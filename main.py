import cv2
import numpy as np
import matplotlib.pyplot as plt

# general constants

# size of ls-piv window and denseness of optical flow
WINDOW_SIZE = 32
STEP = 16
FILENAME = "RMC_D_RC_030_07072020_AC.mp4"

# loop contstants
NUM_TRIALS = 200
FRAME_SEPARATION = 3
PAIR_STEP = 5

# thresholds
CORRELATION_THRESHOLD = 0.2
MAX_DISPLACEMENT_PX = 16
DISPLAY_VECTOR_SCALE = 8

# LS-PIV specific constants from reports (CCC)
MEAN_CHANNEL_WIDTH_M = 10.799
CROSS_SECTION_AREA_M2 = 9.233
#SURFACE_VELOCITY_COEFFICIENT = 0.85
MEASURED_DISCHARGE_M3S = 1.740


def load_video(video_name):
    # open the bubbles video
    cap = cv2.VideoCapture(video_name)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_name}")
    return cap

def select_roi(cap, display_width=1000):
    # read frame and get shape
    ret, frame = cap.read()
    original_height, original_width = frame.shape[:2]
    # use a scale to ensure it doesn't overflow the screen
    scale = display_width / original_width
    display_height = int(original_height * scale)
    resized = cv2.resize(frame, (display_width, display_height))
    # select ROI with scale
    x, y, w, h = cv2.selectROI(resized)
    cv2.destroyAllWindows()

    # convert scaled pixels back to original
    x = int(x / scale)
    y = int(y / scale)
    w = int(w / scale)
    h = int(h / scale)

    return (x, y, w, h)


def get_frame_pair(cap, start_frame, frame_separation, roi):
    # initialize a start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    # read and make sure frame exists
    ret1, frame1 = cap.read()
    if not ret1:
        return None, None
    # control frame separation (testing)
    for _ in range(frame_separation - 1):
        ret, _ = cap.read()
        if not ret:
            return None, None
    # second frame + validation
    ret2, frame2 = cap.read()
    if not ret2:
        return None, None
    # use roi to crop frames
    x, y, w, h = roi
    frame1 = frame1[y:y+h, x:x+w]
    frame2 = frame2[y:y+h, x:x+w]

    return frame1, frame2

def preprocess_frames(frame):
    # start with converting frames to grayscale
    # we'll add more steps as we go along
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    return gray

def get_interrogation_windows(frame, window_size):
    # instantiate windows and find shape
    windows = []
    height, width = frame.shape
    # we want the windows to overlap to form more vectors
    step_size = round(window_size / 2)
    # step over the whole frame and do array slicing to get overlapping windows
    for y in range(0, height - window_size + 1, step_size):
        for x in range(0, width - window_size + 1, step_size):
            window = frame[y:y+window_size,x:x+window_size]
            # remember associated x and y coordinate
            windows.append((window,x,y))
    return windows

def estimate_displacement(windows1, processed2, window_size):
    displacements = []
    # go through all windows in second frame
    for window in windows1:
        # define template and coordinates
        template = window[0]
        x = window[1]
        y = window[2]
        # we want to use the center of the window
        xc = x + window_size // 2
        yc = y + window_size // 2
        
        # define a search region larger than window in frame2
        region_size = window_size * 2
        # find margins so we're looking in a square around window
        margin = (region_size - window_size) // 2
        search_x = x - margin
        search_y = y - margin

        # ignore edges for now
        if search_x < 0 or search_y < 0:
            continue
        if search_x + region_size > processed2.shape[1]:
            continue

        if search_y + region_size > processed2.shape[0]:
            continue
        # get region
        region = processed2[search_y:search_y+region_size, search_x:search_x+region_size]

        # use match template to see where we're the closest to original pattern
        result = cv2.matchTemplate(region,template,cv2.TM_CCOEFF_NORMED)
        # and find max
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        # calculate displacements
        dx = max_loc[0] - margin
        dy = max_loc[1] - margin

        if max_val < CORRELATION_THRESHOLD:
            continue
        if np.sqrt(dx**2 + dy**2) > MAX_DISPLACEMENT_PX:
            continue
        displacements.append((xc,yc,dx,dy,max_val))

    return displacements

def aggregate_displacements(displacements):
    # use dictionary to collect displacements
    grouped = {}

    for xc, yc, dx, dy, corr in displacements:
        key = (xc, yc)
        # use coordinates as key and initialize if doesn't exist
        if key not in grouped:
            grouped[key] = []
        # append displacements/quality at correct coordinate
        grouped[key].append((dx, dy, corr))

    aggregated = []
    # loop over dictionary
    for (xc, yc), values in grouped.items():
        # find values and do statistics
        values = np.array(values)
        median_dx = np.median(values[:, 0])
        median_dy = np.median(values[:, 1])
        median_corr = np.median(values[:, 2])
        # list em
        aggregated.append((xc, yc, median_dx, median_dy, median_corr))

    return aggregated


def summarize_lspiv_results(displacements, fps, frame_separation, roi):
    # find the width in pixels of selected region
    _, _, roi_width_px, _ = roi
    # using given length, we can then calculate meters per pixel
    meters_per_pixel = MEAN_CHANNEL_WIDTH_M / roi_width_px
    # find time change w/ fps and frame separation (num frames between frame1 and frame2)
    dt = frame_separation / fps

    dx = np.array([d[2] for d in displacements])
    dy = np.array([d[3] for d in displacements])
    displacement_px = np.sqrt(dx**2 + dy**2)
    # using displacements and time, find speeds
    surface_speed_mps = displacement_px * meters_per_pixel / dt
    median_surface_speed = np.median(surface_speed_mps)
    mean_surface_speed = np.mean(surface_speed_mps)
    # discharge is a function of surface speed, scaling factor, and area (roughly)
    estimated_discharge = (median_surface_speed * CROSS_SECTION_AREA_M2) # removed * SURFACE_VELOCITY_COEFFICIENT
    # find error vs given value
    percent_error = ((estimated_discharge - MEASURED_DISCHARGE_M3S) / MEASURED_DISCHARGE_M3S * 100)
    # print summary
    print("\nLSPIV summary:")
    print("  fps:", fps)
    print("  meters/pixel:", meters_per_pixel)
    print("  median surface speed (m/s):", median_surface_speed)
    print("  mean surface speed (m/s):", mean_surface_speed)
    print("  estimated discharge (m^3/s):", estimated_discharge)
    print("  measured discharge (m^3/s):", MEASURED_DISCHARGE_M3S)
    print("  percent error:", percent_error)

def summarize_optical_flow_results(displacements, fps, frame_separation, roi):
    _, _, roi_width_px, _ = roi
    # find time and meters per pixel, same as above
    meters_per_pixel = MEAN_CHANNEL_WIDTH_M / roi_width_px
    dt = frame_separation / fps
    # extract values
    dx = np.array([d[2] for d in displacements])
    dy = np.array([d[3] for d in displacements])
    quality = np.array([d[4] for d in displacements])
    # get displacement
    displacement_px = np.sqrt(dx**2 + dy**2)
    # use displacement/time to find speed
    surface_speed_mps = displacement_px * meters_per_pixel / dt
    median_surface_speed = np.median(surface_speed_mps)
    mean_surface_speed = np.mean(surface_speed_mps)
    # and speed + geometry to find discharge
    estimated_discharge = (median_surface_speed * CROSS_SECTION_AREA_M2) # removed * SURFACE_VELOCITY_COEFFICIENT
    percent_error = ((estimated_discharge - MEASURED_DISCHARGE_M3S) / MEASURED_DISCHARGE_M3S * 100)
    # print informative summary
    print("\nOptical Flow summary:")
    print("  fps:", fps)
    print("  meters/pixel:", meters_per_pixel)
    print("  number of vectors:", len(displacements))

    print("\nDisplacement:")
    print("  dx mean:", np.mean(dx))
    print("  dx median:", np.median(dx))
    print("  dy mean:", np.mean(dy))
    print("  dy median:", np.median(dy))

    print("\nQuality:")
    print("  min:", quality.min())
    print("  max:", quality.max())
    print("  mean:", quality.mean())
    print("  median:", np.median(quality))

    print("\nVelocity:")
    print("  median surface speed (m/s):", median_surface_speed)
    print("  mean surface speed (m/s):", mean_surface_speed)

    print("\nDischarge:")
    print("  estimated discharge (m^3/s):", estimated_discharge)
    print("  measured discharge (m^3/s):", MEASURED_DISCHARGE_M3S)
    print("  percent error:", percent_error)


def plot_displacements(frame, displacements):
    # retrieve values
    x = np.array([d[0] for d in displacements])
    y = np.array([d[1] for d in displacements])
    dx = np.array([d[2] for d in displacements])
    dy = np.array([d[3] for d in displacements])
    corr = np.array([d[4] for d in displacements])

    # print informative summary
    print("Number of vectors:", len(displacements))

    print("\ndx:")
    print("  min:", dx.min())
    print("  max:", dx.max())
    print("  mean:", dx.mean())
    print("  median:", np.median(dx))

    print("\ndy:")
    print("  min:", dy.min())
    print("  max:", dy.max())
    print("  mean:", dy.mean())
    print("  median:", np.median(dy))

    print("\nCorrelation:")
    print("  min:", corr.min())
    print("  max:", corr.max())
    print("  mean:", corr.mean())
    print("  median:", np.median(corr))

    # plot the vector map
    plt.figure()
    plt.imshow(frame, cmap="gray")
    # vectors are scaled up
    plt.quiver(x,y,dx * DISPLAY_VECTOR_SCALE,dy * DISPLAY_VECTOR_SCALE,color="red",angles="xy",scale_units="xy",scale=1,width=0.003,)
    plt.title(f"Aggregated displacement vectors ({DISPLAY_VECTOR_SCALE}x display scale)")
    plt.show()


def plot_velocity_field(frame, displacements, fps, frame_separation, roi, title):
    # retrieve vector data
    x = np.array([d[0] for d in displacements])
    y = np.array([d[1] for d in displacements])
    dx = np.array([d[2] for d in displacements])
    dy = np.array([d[3] for d in displacements])

    # convert pixel displacement to physical velocity
    _, _, roi_width_px, _ = roi
    meters_per_pixel = MEAN_CHANNEL_WIDTH_M / roi_width_px
    dt = frame_separation / fps

    displacement_px = np.sqrt(dx**2 + dy**2)
    velocity = displacement_px * meters_per_pixel / dt

    print(f"\n{title} velocity field:")
    print("  Number of vectors:", len(displacements))
    print("  Median velocity:", np.median(velocity), "m/s")
    print("  Mean velocity:", np.mean(velocity), "m/s")

    # make figure
    plt.figure(figsize=(12, 7))

    # show original river image
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    dx_display = dx * DISPLAY_VECTOR_SCALE
    dy_display = dy * DISPLAY_VECTOR_SCALE

    # draw velocity vectors
    q = plt.quiver(x, y, dx_display, dy_display, velocity, cmap="turbo", angles="xy", scale_units="xy", scale=1,)
    

    # colorbar
    plt.colorbar(q, label="Surface velocity (m/s)")

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def ls_piv():
    # main LS-PIV flow

    # start with loading video and getting ROI
    cap = load_video("./videos/" + FILENAME)
    fps = cap.get(cv2.CAP_PROP_FPS)
    roi = select_roi(cap)

    all_displacements = []
    last_processed1 = None
    last_frame = None
    len_displacements = 0
    # loop over specified number of trials
    for trial in range(NUM_TRIALS):
        # pair step is how far in video we move forward each trial
        start_frame = trial * PAIR_STEP
        # get your frames
        frame1, frame2 = get_frame_pair(cap, start_frame=start_frame, frame_separation=FRAME_SEPARATION, roi=roi)

        # ending print
        if frame1 is None or frame2 is None:
            print(f"Stopped at trial {trial}: not enough frames")
            break
        # process frames
        processed1 = preprocess_frames(frame1)
        processed2 = preprocess_frames(frame2)
        # save original for plotting
        last_frame = frame1
        last_processed1 = processed1
        # get interrogation windows
        windows1 = get_interrogation_windows(processed1, WINDOW_SIZE)
        # find displacements
        displacements = estimate_displacement(windows1, processed2, WINDOW_SIZE)
        all_displacements.extend(displacements)
        len_displacements = len(displacements)

        # print(f"Trial {trial + 1}: {len(displacements)} valid vectors")

    print("Total valid vectors:", len(all_displacements))

    # print and show all results
    aggregated_displacements = aggregate_displacements(all_displacements)
    summarize_lspiv_results(aggregated_displacements, fps, FRAME_SEPARATION, roi)
    if last_processed1 is not None:
        plot_displacements(last_processed1, aggregated_displacements)
    if last_frame is not None:
        plot_velocity_field(last_frame,aggregated_displacements,fps,FRAME_SEPARATION,roi,"LS-PIV Surface Velocity Field")

    cap.release()


def get_vectors(frame1, frame2, step=16):
    vectors = []

    # forward and backwards flows
    flow_forward = cv2.calcOpticalFlowFarneback(frame1, frame2, None, pyr_scale=0.5, levels=3, winsize=128, iterations=5, poly_n=5, poly_sigma=1.2, flags=0)
    flow_backward = cv2.calcOpticalFlowFarneback(frame2, frame1, None, pyr_scale=0.5,levels=3, winsize=128, iterations=5, poly_n=5, poly_sigma=1.2, flags=0)

    # get shape and sampling coordinates
    height, width = flow_forward.shape[:2]
    y_coords = np.arange(0, height, step)
    x_coords = np.arange(0, width, step)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    # displacement
    dx = flow_forward[y_grid, x_grid, 0]
    dy = flow_forward[y_grid, x_grid, 1]
    # points in frame 2
    new_x = x_grid.astype(np.float32) + dx
    new_y = y_grid.astype(np.float32) + dy
    # points need to be in image
    valid = ((new_x >= 0) & (new_x < width) & (new_y >= 0) & (new_y < height))

    # backwards flow, using remap because not clean integer values
    back_dx = cv2.remap(flow_backward[:, :, 0], new_x.astype(np.float32), new_y.astype(np.float32), cv2.INTER_LINEAR)
    back_dy = cv2.remap(flow_backward[:, :, 1], new_x.astype(np.float32), new_y.astype(np.float32), cv2.INTER_LINEAR)
    # forward-backward  error
    fb_error = np.sqrt((dx + back_dx)**2 +(dy + back_dy)**2)
    quality = 1 / (1 + fb_error)
    # max displacement filter
    displacement = np.sqrt(dx**2 + dy**2)
    valid &= displacement <= MAX_DISPLACEMENT_PX
    # filter on valid
    valid_y, valid_x = np.where(valid)

    for row, col in zip(valid_y, valid_x):
        # extract values
        vectors.append((x_grid[row, col], y_grid[row, col], dx[row, col], dy[row, col],quality[row, col]))

    return vectors

def optical_flow():
    # main optical flow pipeline

    # start with loading video and getting ROI
    cap = load_video("./videos/" + FILENAME)
    fps = cap.get(cv2.CAP_PROP_FPS)
    roi = select_roi(cap)

    all_displacements = []
    last_processed1 = None
    last_frame = None
    # loop over specified number of trials
    for trial in range(NUM_TRIALS):
        # pair step is how far in video we move forward each trial
        start_frame = trial * PAIR_STEP
        # get your frames
        frame1, frame2 = get_frame_pair(cap, start_frame=start_frame, frame_separation=FRAME_SEPARATION, roi=roi)
        # ending print
        if frame1 is None or frame2 is None:
            print(f"Stopped at trial {trial}: not enough frames")
            break
        # process frames
        processed1 = preprocess_frames(frame1)
        processed2 = preprocess_frames(frame2)
        last_processed1 = processed1
        last_frame = frame1

        # find vectors
        vectors = get_vectors(processed1, processed2)
        all_displacements.extend(vectors)

        # print(f"Trial {trial + 1}: {len(displacements)} valid vectors")

    print("Total valid vectors:", len(all_displacements))

    # print and show all results
    aggregated_displacements = aggregate_displacements(all_displacements)
    summarize_optical_flow_results(aggregated_displacements, fps, FRAME_SEPARATION, roi)
    if last_processed1 is not None:
        plot_displacements(last_processed1, aggregated_displacements)
    if last_frame is not None:
        plot_velocity_field(last_frame,aggregated_displacements,fps,FRAME_SEPARATION,roi,"Optical Flow Surface Velocity Field")

    cap.release()


if __name__ == "__main__":
    ls_piv()
    optical_flow()