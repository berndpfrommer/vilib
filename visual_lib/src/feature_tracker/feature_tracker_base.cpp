/*
 * Feature tracker base class
 * feature_tracker_base.cpp
 *
 * Copyright (c) 2019-2020 Balazs Nagy,
 * Robotics and Perception Group, University of Zurich
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 *  1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "vilib/feature_tracker/feature_tracker_base.h"

#include <opencv2/calib3d.hpp>

namespace vilib {

FeatureTrackerBase::FeatureTrackerBase(const FeatureTrackerOptions & options,
                                       const std::size_t & camera_num) :
  options_(options)
#if FEATURE_TRACKER_ENABLE_ADDITIONAL_STATISTICS
  ,life_stat_("[1]","Feature track life")
#endif /* FEATURE_TRACKER_ENABLE_ADDITIONAL_STATISTICS */
{
  tracks_.resize(camera_num);
  tracked_features_num_.resize(camera_num,0);
  detected_features_num_.resize(camera_num,0);
  dist_type_[0] = INVALID;
  dist_type_[1] = INVALID;
}

FeatureTrackerBase::~FeatureTrackerBase(void) {
}

void FeatureTrackerBase::setDetectorCPU(std::shared_ptr<DetectorBase> & detector,
                                        const std::size_t & camera_id) {
  (void)detector;
  (void)camera_id;
  assert(0 && "This handler should have been overridden");
}

void FeatureTrackerBase::setDetectorGPU(std::shared_ptr<DetectorBaseGPU> & detector,
                                        const std::size_t & camera_id) {
  (void)detector;
  (void)camera_id;
  assert(0 && "This handler should have been overridden");
}

void FeatureTrackerBase::setBestNFeatures(int n) {
  options_.use_best_n_features = n;
}

void FeatureTrackerBase::setMinTracksToDetect(int n) {
  options_.min_tracks_to_detect_new_features = n;
}

void FeatureTrackerBase::removeTracks(const std::vector<std::size_t> & ind_to_remove,
                                      const std::size_t & camera_id) {
  /*
   * Note to future self:
   * the original idea came from SVO: container_helpers.h
   * and it was extended with std::move()
   */
  std::vector<struct FeatureTrack> & source = tracks_[camera_id];
  std::vector<struct FeatureTrack> dest;
  std::size_t start_offset;
  std::size_t end_offset;
  dest.reserve(source.size() - ind_to_remove.size());
  for(std::size_t i=0;i<ind_to_remove.size();++i) {
    start_offset = (i==0)?0:(ind_to_remove[i-1]+1);
    end_offset   = ind_to_remove[i];
    std::move(source.begin()+start_offset,source.begin()+end_offset,std::back_inserter(dest));
  }
  start_offset = (ind_to_remove.size()==0)?0:ind_to_remove[ind_to_remove.size()-1]+1;
  std::move(source.begin() + start_offset,source.end(),std::back_inserter(dest));
  source.swap(dest);
}

void FeatureTrackerBase::addFeature(const std::shared_ptr<Frame> & frame,
                                    const struct FeatureTrack & track) {
  std::size_t i = frame->num_features_++;
  frame->px_vec_.col(i) = Eigen::Vector2d((double)track.cur_pos_[0],(double)track.cur_pos_[1]);
  frame->score_vec_[i] = (double)track.first_score_;
  frame->level_vec_[i] = track.first_level_;
  frame->track_id_vec_[i] = track.track_id_;
}

void FeatureTrackerBase::addFeature(const std::shared_ptr<Frame> & frame,
                                    const int & track_id,
                                    const std::size_t & camera_id) {
  struct FeatureTrack & track = tracks_[camera_id][track_id];
  addFeature(frame,track);
}

#if FEATURE_TRACKER_ENABLE_ADDITIONAL_STATISTICS
void FeatureTrackerBase::showAdditionalStat(bool include_remaining) {
  if(include_remaining) {
    /*
    * Note to future self:
    * as some tracks were not added to the statistics (the remaining tracks
    * that were NOT removed), we do some processing before
    * displaying the statistics.
    * We add all tracks with life > 0 (we ignore the newly detected features)
    */
    for(std::size_t camera_id=0;camera_id<tracks_.size();++camera_id) {
      for(const struct FeatureTrack & track : tracks_[camera_id]) {
        if(track.life_ > 0) {
          life_stat_.add(track.life_);
        }
      }
    }
  }
  life_stat_.display();
}
#endif /* FEATURE_TRACKER_ENABLE_ADDITIONAL_STATISTICS */

void FeatureTrackerBase::reset(void) {
  for(std::size_t c=0;c<tracks_.size();++c) {
    tracks_[c].clear();
    tracked_features_num_[c] = 0;
    detected_features_num_[c] = 0;
  }
}

void FeatureTrackerBase::getDisparity(const double & pivot_ratio,
                                      double & total_avg_disparity) {
  /*
   * Note to future self:
   * we select the nth-element from the ordered disparities,
   * and average this for all cameras.
   * For instance, if pivot_ratio is 0.5, then we select the median
   * disparity from all cameras, and average the medians.
   */
  total_avg_disparity = 0;
  for(std::size_t c=0;c<tracks_.size();++c) {
    std::vector<float> cam_disparities;
    // only take tracks that are long enough (life >= 1)
    // break out, if we find the first with life = 0, because 
    // new tracks to to the end of the vector
    const std::vector<struct FeatureTrack> & cam_tracks = tracks_[c];
    for(std::size_t i=0;i<cam_tracks.size();++i) {
      if(cam_tracks[i].life_ > 0) {
        cam_disparities.push_back(cam_tracks[i].cur_disparity_);
      } else {
        break;
      }
    }
    if(cam_disparities.size()==0) {
      continue;
    }
    // once we have all proper dispariies, select pivot element
    const std::size_t pivot_id = pivot_ratio * cam_disparities.size();
    std::nth_element(cam_disparities.begin(),
                     cam_disparities.begin() + pivot_id,
                     cam_disparities.end());
    total_avg_disparity += cam_disparities[pivot_id];
  }
  total_avg_disparity /= tracks_.size();
}

void FeatureTrackerBase::setIntrinsics(
  int camId, const Intrinsics &in) {
  assert(camId >= 0 && camId < 2);
  const auto intr = in.intrinsics;
  k_matrix_[camId] = cv::Matx33d(intr[0], 0, intr[2],
                                 0, intr[1], intr[3],
                                 0,       0,       1);
  dist_type_[camId] =
    (in.dist_type == "equidistant" || in.dist_type == "fisheye") ? EQUIDIST : RADTAN;
  dist_coeff_[camId] = in.dist_coeffs;
  resolution_[camId] = in.resolution;
  if (dist_type_[0] != INVALID && dist_type_[1] != INVALID &&
      extrinsics_valid_) {
    calibration_valid_ = true;
  }
}

void FeatureTrackerBase::setExtrinsics(const Eigen::Isometry3d &T) {
  rect_matrix_ = cv::Matx33d(T(0, 0), T(0, 1), T(0, 2),
                             T(1, 0), T(1, 1), T(1, 2),
                             T(2, 0), T(2, 1), T(2, 2));
  extrinsics_valid_ = true;
  if (dist_type_[0] != INVALID && dist_type_[1] != INVALID &&
      extrinsics_valid_) {
    calibration_valid_ = true;
  }
}

void FeatureTrackerBase::transformPointsToCam1(
  const std::vector<FeatureTrack> &tracks,
  std::vector<cv::Point2f> *cam1_points) {
  // fetch cam0 points
  std::vector<cv::Point2f> cam0Points(tracks.size());
  for (size_t i = 0; i < tracks.size(); i++) {
    const auto &track = tracks[i];
    cam0Points[i].x = track.cur_pos_[0];
    cam0Points[i].y = track.cur_pos_[1];
  }
  // undistort cam0 points and use rotation only to tranform
  // to cam1. This is correct only for points infinitely far away
  std::vector<cv::Point2f> cam1UndistPoints; // temp array
  switch (dist_type_[0]) {
  case EQUIDIST:
    cv::fisheye::undistortPoints(cam0Points, cam1UndistPoints,
                                 k_matrix_[0], dist_coeff_[0],
                                 rect_matrix_);
    break;
  case RADTAN:
  default:
    cv::undistortPoints(cam0Points, cam1UndistPoints,
                        k_matrix_[0], dist_coeff_[0],
                        rect_matrix_);
  }
  // distort points to cam1
  switch (dist_type_[1]) {
  case EQUIDIST:
    cv::fisheye::distortPoints(cam1UndistPoints, *cam1_points, k_matrix_[1], dist_coeff_[1]);
    break;
  default:
  case RADTAN: {
    std::vector<cv::Point3f> cam1UndistPointsH;
    cv::convertPointsToHomogeneous(cam1UndistPoints, cam1UndistPointsH);
    cv::projectPoints(cam1UndistPointsH, cv::Vec3d::zeros(), cv::Vec3d::zeros(),
                      k_matrix_[1], dist_coeff_[1], *cam1_points);
    break; }
  }
}
} // namespace vilib
