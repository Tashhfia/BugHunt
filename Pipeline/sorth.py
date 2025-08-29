"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

np.random.seed(0)


def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
  return(o)  


def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):
    """
    Internal state of one tracked object (7-state constant-velocity model)
      x  = cx, cy, area, ratio, vx, vy, d_area
      z  = cx, cy, area, ratio
    """
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # ── constant-velocity state-transition (dt = 1 frame) ──
        dt = 1.0
        self.kf.F = np.array(
            [[1,0,0,0,dt,0 ,0 ],
             [0,1,0,0,0 ,dt,0 ],
             [0,0,1,0,0 ,0 ,dt],
             [0,0,0,1,0 ,0 ,0 ],
             [0,0,0,0,1 ,0 ,0 ],
             [0,0,0,0,0 ,1 ,0 ],
             [0,0,0,0,0 ,0 ,1 ]], dtype=float)

        self.kf.H = np.array(
            [[1,0,0,0,0,0,0],
             [0,1,0,0,0,0,0],
             [0,0,1,0,0,0,0],
             [0,0,0,1,0,0,0]], dtype=float)

        # ── measurement noise (YOLO jitter) ──
        px_sigma, py_sigma = 2.0, 2.0          # pixels
        area_sigma         = 400.0             # pixel²
        ratio_sigma        = 0.05              # unit-less
        self.kf.R = np.diag([px_sigma**2, py_sigma**2,
                             area_sigma**2, ratio_sigma**2])

        # ── process noise (how much a bug can accelerate per frame) ──
        q_pos  = 4.0                           # px²
        q_vel  = 10.0                          # (px/frame)²
        q_area = 200.0                         # pixel²
        q_dA   = 400.0                         # pixel²/frame
        self.kf.Q = np.diag([q_pos, q_pos, q_area, 0.02,
                             q_vel, q_vel, q_dA])

        # ── initial uncertainty ──
        self.kf.P[4:, 4:] *= 1000.             # large for unobserved v
        self.kf.P *= 10.0

        # initialise state with first bbox
        self.kf.x[:4] = convert_bbox_to_z(bbox)

        # book-keeping
        self.time_since_update = 0
        self.id      = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []          # stores last predict() result
        self.hits    = 0
        self.hit_streak = 0
        self.age     = 0

    # ------------------------------------------------------------------ #
    def current_speed(self):
        vx, vy = float(self.kf.x[4]), float(self.kf.x[5])
        return (vx**2 + vy**2) ** 0.5

    def update(self, bbox):
        """Correct state with an observed bbox."""
        self.time_since_update = 0
        self.hits      += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

        # ── velocity boot-strap after the 2nd hit ──
        if self.hits == 2:
            cx_prev, cy_prev = self.history[-1][0, 0], self.history[-1][0, 1]
            cx_now  = bbox[0] + (bbox[2] - bbox[0]) / 2
            cy_now  = bbox[1] + (bbox[3] - bbox[1]) / 2
            self.kf.x[4] = cx_now - cx_prev      # vx
            self.kf.x[5] = cy_now - cy_prev      # vy

        self.history = []           # clear after correction

    def predict(self):
        """Advance the state one frame and return predicted bbox."""
        # keep d_area non-negative
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] = 0.0

        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        bbox = convert_x_to_bbox(self.kf.x)
        self.history.append(bbox)
        return bbox

    def get_state(self):
        """Return the current bounding box estimate."""
        return convert_x_to_bbox(self.kf.x)



def associate_detections_to_trackers(detections, trackers, dist_threshold=50):
    """
    Match detections to trackers purely by centroid distance.
      detections : N×5  [x1,y1,x2,y2,score]
      trackers   : M×5  [x1,y1,x2,y2,0]
    Returns (matches, unmatched_dets, unmatched_trks)
    """
    if len(trackers) == 0:
        return (np.empty((0, 2), int),
                np.arange(len(detections)),
                np.empty((0, 2), int))

    # ---------- centroids & distance matrix ----------
    det_centers = np.column_stack([(detections[:, 0] + detections[:, 2]) / 2,
                                   (detections[:, 1] + detections[:, 3]) / 2])
    trk_centers = np.column_stack([(trackers[:, 0] + trackers[:, 2]) / 2,
                                   (trackers[:, 1] + trackers[:, 3]) / 2])

    dist_matrix = np.linalg.norm(det_centers[:, None, :] -
                                 trk_centers[None, :, :], axis=2)

    # ---------- Hungarian assignment ----------
   
    row_idx, col_idx = linear_sum_assignment(dist_matrix)
    matches = np.array(list(zip(row_idx, col_idx)))

    # ---------- gate check (scalar or per-tracker vector) ----------
    if np.ndim(dist_threshold) == 0:          # single value
        gate_vec = np.full(trackers.shape[0], dist_threshold, dtype=float)
    else:                                     # already a vector (length M)
        gate_vec = np.asarray(dist_threshold, dtype=float)

    good = dist_matrix[row_idx, col_idx] <= gate_vec[col_idx]
    matches = matches[good]

    # ---------- build unmatched lists ----------
    if matches.size == 0:
        return (matches.reshape(0, 2),
                np.arange(len(detections)),
                np.arange(len(trackers)))

    all_dets = np.arange(len(detections))
    all_trks = np.arange(len(trackers))
    unmatched_dets = np.setdiff1d(all_dets, matches[:, 0])
    unmatched_trks = np.setdiff1d(all_trks, matches[:, 1])

    return matches, unmatched_dets, unmatched_trks



class Sort(object):
  def __init__(self, max_age=1, min_hits=3, dist_threshold=70, k_speed = 2.0,  # extra gate / (pixel·frame-1)
                 max_gate= 220, max_ids = None ):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.dist_threshold = dist_threshold
    self.k_speed        = k_speed
    self.max_gate       = max_gate
    self.trackers = []
    self.dead_pool  = []  # for dead trackers that are not yet removed
    self.frame_count = 0
    self.max_ids = max_ids  # maximum number of IDs to track
    self.total_ids = 0  # total number of IDs assigned so far

  def update(self, dets: np.ndarray = np.empty((0, 5))):
      """
      Run one tracker step.

      Args
      ----
      dets : ndarray (N×5)    detections [x1,y1,x2,y2,score]
                              pass np.empty((0,5)) for frames with no detections

      Returns
      -------
      ndarray (M×5)           [x1,y1,x2,y2,id] for every track (confirmed or coasting)
      """
      # ───────────────────────────────────────────────── warm-up / ceiling ──
      self.frame_count += 1
      if self.frame_count <= 2:                     # warm-up window
          if self.max_ids is None:
              self.max_ids = 0
          self.max_ids = max(self.max_ids, len(dets))

      # ───────────────────────────────────── predict all live trackers ─────
      trks   = np.zeros((len(self.trackers), 5))
      to_del = []
      for t, slot in enumerate(trks):
          pos = self.trackers[t].predict()[0]
          slot[:] = [pos[0], pos[1], pos[2], pos[3], 0]
          if np.any(np.isnan(pos)):
              to_del.append(t)

      trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
      for t in reversed(to_del):
          self.trackers.pop(t)

      # ────────────────────────────── build adaptive centroid gates ────────
      gates = []
      for trk in self.trackers:
          speed = trk.current_speed()
          coast = trk.time_since_update
          g     = self.dist_threshold + self.k_speed * speed * coast
          gates.append(min(g, self.max_gate))
      gates = np.asarray(gates)

      # ─────────────────────────────── assign detections → trackers ────────
      matched, unmatched_dets, _ = associate_detections_to_trackers(dets, trks, gates)

      # update matched
      for d_idx, t_idx in matched:
          self.trackers[t_idx].update(dets[d_idx])

      # ───────────────────── recycle or create trackers for leftovers ──────
      for d_idx in unmatched_dets:
          if self.dead_pool:                         # reuse an old ID
              _, trk = self.dead_pool.pop(0)         # keep its velocity!
              trk.update(dets[d_idx])                # ← NO re-initialise
                                                   #   (ID already correct)
          elif self.total_ids < self.max_ids:        # brand-new ID
              trk = KalmanBoxTracker(dets[d_idx])
              self.total_ids += 1
          else:
              continue                               # ceiling reached
          self.trackers.append(trk)

      # ───────────────────────── emit boxes & cull stale trackers ──────────
      ret = []
      for idx in range(len(self.trackers) - 1, -1, -1):   # iterate backwards → safe pop
          trk = self.trackers[idx]
          d   = trk.get_state()[0]

          ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))

          # retire tracker if it exceeded max_age
          if trk.time_since_update > self.max_age:
              self.dead_pool.append((self.frame_count, trk))
              self.trackers.pop(idx)

      # keep dead_pool ≤ 120 frames old
      self.dead_pool = [(f, t) for f, t in self.dead_pool
                        if self.frame_count - f < 120]

      return np.concatenate(ret) if ret else np.empty((0, 5))

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=1)
    parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
  # all train
  args = parse_args()
  display = args.display
  phase = args.phase
  total_time = 0.0
  total_frames = 0
  colours = np.random.rand(32, 3) #used only for display
  if(display):
    if not os.path.exists('mot_benchmark'):
      print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
      exit()
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')

  if not os.path.exists('output'):
    os.makedirs('output')
  pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')
  for seq_dets_fn in glob.glob(pattern):
    mot_tracker = Sort(max_age=args.max_age, 
                       min_hits=args.min_hits) #create instance of the SORT tracker
    seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
    seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]
    
    with open(os.path.join('output', '%s.txt'%(seq)),'w') as out_file:
      print("Processing %s."%(seq))
      for frame in range(int(seq_dets[:,0].max())):
        frame += 1 #detection and frame numbers begin at 1
        dets = seq_dets[seq_dets[:, 0]==frame, 2:7]
        dets[:, 2:4] += dets[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
        total_frames += 1

        if(display):
          fn = os.path.join('mot_benchmark', phase, seq, 'img1', '%06d.jpg'%(frame))
          im =io.imread(fn)
          ax1.imshow(im)
          plt.title(seq + ' Tracked Targets')

        start_time = time.time()
        trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        for d in trackers:
          print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)
          if(display):
            d = d.astype(np.int32)
            ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:]))

        if(display):
          fig.canvas.flush_events()
          plt.draw()
          ax1.cla()

  print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

  if(display):
    print("Note: to get real runtime results run without the option: --display")
