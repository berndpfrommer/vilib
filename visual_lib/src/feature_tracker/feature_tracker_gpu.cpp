/*
 * Feature tracker GPU
 * feature_tracker_gpu.cpp
 *
 * Copyright (c) 2019-2020 Balazs Nagy,
 * Robotics and Perception Group, University of Zurich
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
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

#include <numeric>
#include "vilib/feature_tracker/feature_tracker_gpu.h"
#include "vilib/feature_tracker/feature_tracker_options.h"
#include "vilib/feature_tracker/feature_tracker_cuda_tools.h"
#include "vilib/common/point.h"
#include "vilib/cuda_common.h"

namespace vilib {

#define VERBOSE_TRACKING                      0
/*
 * Note to future self:
 * we rely on this byte size in various parts of the code
 */
#define METADATA_ELEMENT_BYTES                64

FeatureTrackerGPU::FeatureTrackerGPU(const FeatureTrackerOptions & options,
                                     const std::size_t & camera_num) :
  FeatureTrackerBase(options,camera_num),
  pyramid_levels_(options_.klt_max_level - options_.klt_min_level + 1) {
  assert(options_.klt_max_level > options_.klt_min_level &&
         "Maximum level should exceed minimum level");
  // Buffer pointers
  buffer_.resize(camera_num);
  for(std::size_t c=0;c<camera_num; ++c) {
    buffer_[c].h_indir_data_ = nullptr;
    buffer_[c].h_metadata_   = nullptr;
    buffer_[c].d_patch_data_ = nullptr;
    buffer_[c].d_hessian_data_ = nullptr;
  }
  // Detectors, streams
  detector_.resize(camera_num,nullptr);
  stream_.resize(camera_num,nullptr);
  // Search for maximum patch size, including borders
  int max_patch_size = 0;
  for(std::size_t i=0;i<options_.klt_patch_sizes.size();++i) {
    pyramid_patch_sizes_.wh[i] = options_.klt_patch_sizes[i];
    if(options_.klt_patch_sizes[i] > max_patch_size) {
      max_patch_size = options_.klt_patch_sizes[i];
    }
  }
  pyramid_patch_sizes_.max_area = (max_patch_size+2)*(max_patch_size+2);
}

FeatureTrackerGPU::~FeatureTrackerGPU(void) {
  for(std::size_t c=0;c<buffer_.size();++c) {
    freeStorage(c);
  }
}

#if 0
void FeatureTrackerGPU::trackFeatures(
  PyramidInfo *prev_pyr,
  PyramidInfo *cur_pyr,
  const std::vector<std::vector<Feature>> &prev_features,
  std::vector<std::vector<Feature>> *cur_features,
  size_t cam_id) {
  assert(prev_pyr.base_frames.size() >= cam_id &&
         cur_pyr.base_frames.size() >= cam_id &&
         prev_pyr.pyramid_descs.size() >= cam_id &&
         cur_pyr.pyramid_descs.size() >= cam_id &&
         detector_.size() >= cam_id &&
         "cam_id out of range");
  // clear all tracks before detection
  std::vector<std::shared_ptr<Frame>> cur_base_frames;

  // remove all features from frame and release track
  // memory
  clearTracksAndFeatures(&cur_base_frames[c], &buffer_[c],
                         &tracks_[c], &tracked_features_num_[c]);
  addFeaturesToTracks(cur_pyr->base_frames[c], cur_pyr->pyramid_descs,
                      prev_features, &detected_features_num_[c]);
  (void) cur_features;

}
#endif


void FeatureTrackerGPU::track(const std::shared_ptr<FrameBundle> & cur_frames,
                              std::size_t & total_tracked_features_num,
                              std::size_t & total_detected_features_num) {
  assert(cur_frames->size() == detector_.size() &&
         "The frame count and the detector count differs");
#if VERBOSE_TRACKING
  static std::size_t frame_bundle_id = 1;
  std::cout << "Frame bundle " << (frame_bundle_id++) << " -------------" << std::endl;
#endif /* VERBOSE_TRACKING */

  // check for detector being set up
  for(std::size_t c=0;c<cur_frames->size();++c) {
    assert(detector_[c] != nullptr && "One must set a GPU feature detector first");
  }
  // 00) Prerequisites
  PyramidInfo pyInfo;
  computePyramidInfo(&pyInfo, cur_frames);
  std::vector<image_pyramid_descriptor_t> &cur_pyramids = pyInfo.pyramid_descs;
  std::vector<std::shared_ptr<Frame>> &cur_base_frames = pyInfo.base_frames;

  // 01) Track existing feature tracks (only if there are any)
  for(std::size_t c=0;c<cur_frames->size();++c) {
    trackOnGPU(cur_pyramids[c], pyramid_patch_sizes_, tracks_[c],
               &buffer_[c], &stream_[c]);
  }

  // 02) Process Tracking results
  for(std::size_t c=0;c<cur_frames->size();++c) {
    processTrackingResults(cur_base_frames[c], &tracks_[c],
                           &tracked_features_num_[c],
                           &buffer_[c], &stream_[c]);
  }

  // 03) Detect new features (if necessary)
  std::vector<std::vector<Feature>> features_found(cur_frames->size());
  for(std::size_t c=0;c<cur_frames->size();++c) {
    if(tracked_features_num_[c] < options_.min_tracks_to_detect_new_features) {
      if(options_.reset_before_detection) {
        // clear all tracks before detection
        clearTracksAndFeatures(cur_base_frames[c], &buffer_[c],
                               &tracks_[c], &tracked_features_num_[c]);
      }
      const size_t max_new_features = (size_t)std::max(
        0, options_.use_best_n_features - (int)tracked_features_num_[c]);
      detectNewFeatures(&features_found[c], *cur_base_frames[c],
                        max_new_features, detector_[c]);
    }
    // add new features to tracks
    addFeaturesToTracks(cur_base_frames[c], cur_pyramids[c], features_found[c],
                        tracked_features_num_[c],
                        &buffer_[c], &detected_features_num_[c], &tracks_[c],
                        &stream_[c]);
#if VERBOSE_TRACKING
    std::cout << " Detected features (cam_id=" << c << "): " << detected_features_num_[c] << std::endl;
#endif /* VERBOSE_TRACKING */

  }
  // 04) Do the accumulation for statistics
  total_tracked_features_num = std::accumulate(tracked_features_num_.begin(),
                                               tracked_features_num_.end(),
                                               0);
  total_detected_features_num = std::accumulate(detected_features_num_.begin(),
                                                detected_features_num_.end(),
                                                0);
}

void FeatureTrackerGPU::computePyramidInfo(
  PyramidInfo *pyr,
  const std::shared_ptr<FrameBundle> & frames) {
  for(std::size_t c=0;c<frames->size();++c) {
    pyr->base_frames.push_back(frames->at(c));
    pyr->pyramid_descs.push_back(pyr->base_frames.back()->getPyramidDescriptor());
    pyr->base_frames.back()->resizeFeatureStorage(max_ftr_count_);
  }
}


void FeatureTrackerGPU::processTrackingResults(
  const std::shared_ptr<Frame> &base_frame,
  std::vector<FeatureTrack> *tracks,
  size_t *tracked_features_num,
  struct GPUBuffer *buffer_ptr,
  cudaStream_t *stream_ptr)
{
  auto &buffer = *buffer_ptr;
  *tracked_features_num = 0;
  std::vector<std::size_t> remove_indices;
  if(tracks->size() > 0) {
    // Just synchronize with the stream..
    CUDA_API_CALL(cudaStreamSynchronize(*stream_ptr));
    // And check the results
    for(std::size_t i=0; i<tracks->size(); ++i) {
      struct FeatureTrack & track = (*tracks)[i];
      std::size_t offset = track.buffer_id_*METADATA_ELEMENT_BYTES/4;
      // if out px_x is nan, then remove the track, otherwise, update its current location
      float x = buffer.h_cur_px_[offset];
      if(std::isnan(x)) {
        // we didnt converge in the KLT tracker
        remove_indices.push_back(i);
        releaseBufferId(buffer_ptr, track.buffer_id_);
#if FEATURE_TRACKER_ENABLE_ADDITIONAL_STATISTICS
        life_stat_.add(track.life_);
#endif /* FEATURE_TRACKER_ENABLE_ADDITIONAL_STATISTICS */
      } else {
        ++track.life_;
        // Last Position
        float y = buffer.h_cur_px_[offset+1];
        track.cur_pos_[0] = x;
        track.cur_pos_[1] = y;
        // New disparity
        track.cur_disparity_ = buffer.h_cur_disparity_[offset];
        // Update reference frame?
        if(options_.klt_template_is_first_observation == false) {
          // store shared pointer, so the frame is not freed
          track.template_frame_ = base_frame;
          // update GPU template position
          buffer.h_template_px_[offset] = x;
          buffer.h_template_px_[offset+1] = y;
        }
        ++(*tracked_features_num);
        addFeature(base_frame, track);
      }
    }
  }
#if VERBOSE_TRACKING
  std::cout << " Tracked features (cam_id=" << c << "): " << *tracked_features_num << std::endl;
#endif /* VERBOSE_TRACKING */
  // Remove terminated tracks
#if VERBOSE_TRACKING
  std::cout << " Tracked features lost (cam_id=" << c << "): " << remove_indices.size() << std::endl;
#endif /* VERBOSE_TRACKING */
  removeTracks(remove_indices, tracks);
}
                                       
void  FeatureTrackerGPU::trackOnGPU(
  const image_pyramid_descriptor_t & pyramid_description,
  const pyramid_patch_descriptor_t & pyramid_patch_sizes,
  const std::vector<FeatureTrack> &tracks,
  struct GPUBuffer *buffer_ptr,
  cudaStream_t *stream_ptr) {
  auto &buffer = *buffer_ptr;
  if(tracks.size() > 0) {
    for(std::size_t track_id=0; track_id<tracks.size(); ++track_id) {
      // opposed to the previous version, now we only need to update
      // the indirection layer that points to the already occupied buffer cells
      buffer.h_indir_data_[track_id] = tracks[track_id].buffer_id_;
    }
    // run the tracking on the GPU
    feature_tracker_cuda_tools::track_features(options_.affine_est_offset,
                                               options_.affine_est_gain,
                                               tracks.size(),
                                               options_.klt_min_level,
                                               options_.klt_max_level,
                                               options_.klt_min_update_squared,
                                               pyramid_description,
                                               pyramid_patch_sizes,
                                               buffer.d_indir_data_,
                                               buffer.d_patch_data_,
                                               buffer.d_hessian_data_,
                                               buffer.d_first_px_,
                                               buffer.d_cur_px_,
                                               buffer.d_cur_alpha_beta_,
                                               buffer.d_cur_f_,
                                               buffer.d_cur_disparity_,
                                               *stream_ptr);
  }
}

void FeatureTrackerGPU::clearTracksAndFeatures(
  const std::shared_ptr<Frame> &base_frame, GPUBuffer *buffer,
  std::vector<FeatureTrack> *tracks, size_t *tracked_features_num) {
  *tracked_features_num = 0;
  // free the buffers first
  for(auto track : *tracks) {
    releaseBufferId(buffer, track.buffer_id_);
  }
  tracks->clear();
  // discard all previously added features from tracking
  base_frame->num_features_ = 0;
}


void FeatureTrackerGPU::addFeaturesToTracks(
  const std::shared_ptr<Frame> &base_frame,
  const image_pyramid_descriptor_t &pyramid,
  const std::vector<Feature> &features,
  size_t tracked_features_num,
  GPUBuffer *buffer,
  size_t *detected_features_num,
  std::vector<FeatureTrack> *tracks,
  cudaStream_t *stream_ptr) {
  // add new features to track
  *detected_features_num = 0;
  for (const auto f: features) {
    int track_index = addTrack(buffer, tracks, base_frame, f.x, f.y, f.level, f.score);
    // Add tracked point to output
    addFeature(base_frame, (*tracks)[track_index]);
    ++(*detected_features_num);
  }
  // 04) Precomopute patches & Hessians
  if(options_.klt_template_is_first_observation) {
    // only precompute the newly detected features
    updateTracks(*detected_features_num, pyramid, buffer, tracks, stream_ptr);
  } else {
    // precompute all current feature tracks
    updateTracks(*detected_features_num + tracked_features_num, pyramid, buffer, tracks, stream_ptr);
  }
}

void FeatureTrackerGPU::detectNewFeatures(
  std::vector<Feature> *features,
  const Frame &base_frame,
  size_t max_new_features,
  const std::shared_ptr<DetectorBaseGPU> &detector) {

  OccupancyGrid2D & detector_grid = detector->getGrid();
  detector_grid.reset();
  // mark used grid cells as occupied so the detector does
  // not find new features there
  for(std::size_t i=0;i<base_frame.num_features_;++i) {
    const Eigen::Vector2d & pos_2d = base_frame.px_vec_.col(i);
    detector_grid.setOccupied(pos_2d[0],pos_2d[1]);
  }
  /*
   * Note to future self:
   * pre-occupied cells will be isEmpty(cell_index) = false
   */
  features->reserve(detector_grid.size());
  detector->detect(
    base_frame.pyramid_,
    [&](const std::size_t & grid_cell_cnt,
        const float * h_pos,
        const float * h_score,
        const int   * h_level) {
          // Do we use every detected feature?
          if(options_.use_best_n_features == -1) {
            // Yes, we are using every feature
            for(std::size_t cell_index=0;cell_index<detector_grid.size();++cell_index) {
              if(detector_grid.isEmpty(cell_index) && h_score[cell_index] > 0.0f) {
                features->push_back(Feature(h_pos[cell_index * 2],
                                            h_pos[cell_index * 2 + 1],
                                            h_level[cell_index],
                                            h_score[cell_index]));
              }
            }
          } else {
            // No, we are only using the best N features according to their score
            std::vector<std::size_t> idx(grid_cell_cnt);
            std::iota(idx.begin(), idx.end(), 0u);
            std::sort(idx.begin(), idx.end(), [&](std::size_t i1, std::size_t i2) {
              return h_score[i1] > h_score[i2];
            });
            for(std::size_t i=0;
                i<detector_grid.size() && features->size() < max_new_features;
                ++i) {
              std::size_t cell_index = idx[i];
              if(detector_grid.isEmpty(cell_index) && h_score[cell_index] > 0.0f) {
                features->push_back(Feature(h_pos[cell_index * 2],
                                            h_pos[cell_index * 2 + 1],
                                            h_level[cell_index],
                                            h_score[cell_index]));
              }
            }
          }
    });

}

void FeatureTrackerGPU::updateTracks(const std::size_t & last_n,
                                     const image_pyramid_descriptor_t & pyramid_description,
                                     GPUBuffer *buffer,
                                     std::vector<FeatureTrack> *tracks,
                                     cudaStream_t *stream_ptr) {
  if(last_n == 0) {
    return;
  }

  std::size_t offset = tracks->size() - last_n;
  // initialize the indirection layer with used buffer ids
  std::size_t indir_id = 0;
  for(auto it=tracks->begin() + offset; it < tracks->end(); ++it) {
    buffer->h_indir_data_[indir_id++] = it->buffer_id_;
  }

  // start a kernel for initializing the templates and the inverse hessians
  feature_tracker_cuda_tools::update_tracks(last_n,
                                            options_.affine_est_offset,
                                            options_.affine_est_gain,
                                            options_.klt_min_level,
                                            options_.klt_max_level,
                                            pyramid_description,
                                            pyramid_patch_sizes_,
                                            buffer->d_indir_data_,
                                            buffer->d_template_px_,
                                            buffer->d_patch_data_,
                                            buffer->d_hessian_data_,
                                            *stream_ptr);
}

int FeatureTrackerGPU::addTrack(
  GPUBuffer *buffer,
  std::vector<FeatureTrack> *tracks,
  const std::shared_ptr<Frame> & first_frame,
  const float & first_x,
  const float & first_y,
  const int & first_level,
  const float & first_score) {
  // Acquire free buffer id
  int buffer_id = acquireBufferId(buffer);
  // Acquire new track id
  int track_id = Point::getNewId();
  // Create new feature track struct
  tracks->emplace_back(first_frame,
                       first_x,
                       first_y,
                       first_level,
                       first_score,
                       track_id,
                       buffer_id);
  // Update buffer elements
  std::size_t offset = buffer_id*METADATA_ELEMENT_BYTES/4;
  // Update ref_px   = {first_x,first_y}
  buffer->h_template_px_[offset] = first_x;
  buffer->h_template_px_[offset+1] = first_y;
  // Update first_px = {first_x,first_y}
  buffer->h_first_px_[offset] = first_x;
  buffer->h_first_px_[offset+1] = first_y;
  // Update search_px = {first_x,first_y}
  buffer->h_cur_px_[offset] = first_x;
  buffer->h_cur_px_[offset+1] = first_y;
  // Update alpha-beta = {0.0,0.0}
  buffer->h_cur_alpha_beta_[offset] = 0.0f;
  buffer->h_cur_alpha_beta_[offset+1] = 0.0f;
  return tracks->size()-1;
}

void FeatureTrackerGPU::setDetectorGPU(std::shared_ptr<DetectorBaseGPU> & detector,
                                       const std::size_t & camera_id) {
  // Free previous storage buffers if any
  freeStorage(camera_id);
  // Get detector, stream
  detector_[camera_id] = detector;
  stream_[camera_id] = detector->getStream();
  /*
   * Note to future self:
   * as at this point we already know the grid size, allocate the maximum number of
   * features to be tracked
   */
  /*
   * Note to future self:
   * 3 parts:
   * Layer of indirection: Array of integers, that point to AoS blocks [mapped memory]
   * Metadata elements: Array of structs [mapped memory]
   * Patchdata: array of structs [device memory]
   *
   * ----------------------------------------------------------------
   * Layer of indirection: consecutively placed ids point to buffers
   * int: 4 bytes
   * ----------------------------------------------------------------
   * Metadata: -> data stays in places
   * (16 + 8 + 8 + 8 + 8 + 4) = 52 bytes + 12 padding bytes = 64 bytes
   *
   * -> this could be either AoS or SoA
   * for the GPU-> since each warp uses exactly 1 parameter, it doesnt matter
   * for the CPU-> I think since the elements are closer, and we access 1 element, and then
   *               we jump to the next one, I'd assume AoS is faster
   *
   * [current]  search_bearing_vector: float (4x 4 bytes) [updated by the kernel]
   * [template] ref_px: float (2x 4 bytes) -> for patch preloading (can be the first or the current)
   * [first]    first_px: float (2x 4 bytes) -> for disparity calculation
   * [current]  search_px: float (2x 4 bytes) [updated by the kernel]
   * [current]  search_alpha_beta: float (2x 4 bytes) [updated by the kernel]
   * [current]  disparity: float (1x 4 bytes) [updated by the kernel]
   * ----------------------------------------------------------------
   * Patch data:
   * [template] patch level max_level
   * [template] patch level max_level-1
   * ...
   * [template] patch level min_level
   * ----------------------------------------------------------------
   * Hessian:
   * [template] inverse hessian level max_level
   * [template] inverse hessian level max_level-1
   * ...
   * [template] inverse hessian level min_level
   */
  std::size_t max_detected_ftr_count = 
    (options_.use_best_n_features==-1)?detector_[camera_id]->getGrid().size():options_.use_best_n_features;
  /*
   * Note to future self:
   * worst case scenario: all tracked features group in 1 cell, and their count drops below
   * min_tracks_to_detect_new_features, and then we detect (max_detected_ftr_count-1)
   */
  max_ftr_count_ = (options_.min_tracks_to_detect_new_features-1) + (max_detected_ftr_count-1);

  // 00) Create available indices for buffer bins
  // Note: the elements should be in decreasing order
  initBufferIds(camera_id);

  // 01) Allocate indirection layer
  CUDA_API_CALL(cudaHostAlloc((void**)&buffer_[camera_id].h_indir_data_,(max_ftr_count_ * sizeof(int)),cudaHostAllocMapped));
  CUDA_API_CALL(cudaHostGetDevicePointer((void**)&buffer_[camera_id].d_indir_data_,buffer_[camera_id].h_indir_data_,0));

  // 02) Metadata
  std::size_t metadata_bytes = (max_ftr_count_ * METADATA_ELEMENT_BYTES);
  // allocate mapped memory (as seen during feature alignment, it is going to be
  // faster, as we mainly only access 1 address once and then write out the result
  CUDA_API_CALL(cudaHostAlloc((void**)&buffer_[camera_id].h_metadata_,metadata_bytes,cudaHostAllocMapped));
  CUDA_API_CALL(cudaHostGetDevicePointer((void**)&buffer_[camera_id].d_metadata_,buffer_[camera_id].h_metadata_,0));

  // 03) Patchdata
  std::size_t patchdata_element_bytes = sizeof(FEATURE_TRACKER_REFERENCE_PATCH_TYPE)*pyramid_patch_sizes_.max_area*pyramid_levels_;
  std::size_t patchdata_bytes = patchdata_element_bytes * max_ftr_count_;
  // device memory, as the CPU doesnt need to access it
  CUDA_API_CALL(cudaMalloc((void**)&buffer_[camera_id].d_patch_data_,patchdata_bytes));

  // 04) Hessian data
  std::size_t hessian_element_bytes = sizeof(float)*10*pyramid_levels_;
  std::size_t hessian_bytes = hessian_element_bytes * max_ftr_count_;
  // device memory, as the CPU doesnt need to access it
  CUDA_API_CALL(cudaMalloc((void**)&buffer_[camera_id].d_hessian_data_,hessian_bytes));

  // Adjust device pointers
  unsigned char * d_ptr = buffer_[camera_id].d_metadata_;
  unsigned char * h_ptr = buffer_[camera_id].h_metadata_;
  // Bearing vector
  buffer_[camera_id].d_cur_f_ = (float4*)d_ptr; // 16-byte aligned
  buffer_[camera_id].h_cur_f_ = (float*)h_ptr;
  d_ptr += sizeof(float)*4;
  h_ptr += sizeof(float)*4;
  // Template position
  buffer_[camera_id].d_template_px_ = (float2*)d_ptr; // 8-byte aligned
  buffer_[camera_id].h_template_px_ = (float*)h_ptr;
  d_ptr += sizeof(float)*2;
  h_ptr += sizeof(float)*2;
  // First position
  buffer_[camera_id].d_first_px_ = (float2*)d_ptr; // 8-byte aligned
  buffer_[camera_id].h_first_px_ = (float*)h_ptr;
  d_ptr += sizeof(float)*2;
  h_ptr += sizeof(float)*2;
  // Current position
  buffer_[camera_id].d_cur_px_ = (float2*)d_ptr; // 8-byte aligned
  buffer_[camera_id].h_cur_px_ = (float*)h_ptr;
  d_ptr += sizeof(float)*2;
  h_ptr += sizeof(float)*2;
  // Current alpha-beta
  buffer_[camera_id].d_cur_alpha_beta_ = (float2*)d_ptr; // 8-byte aligned
  buffer_[camera_id].h_cur_alpha_beta_ = (float*)h_ptr;
  d_ptr += sizeof(float)*2;
  h_ptr += sizeof(float)*2;
  // Current disparity
  buffer_[camera_id].d_cur_disparity_ = (float*)d_ptr; // 8-byte aligned
  buffer_[camera_id].h_cur_disparity_ = (float*)h_ptr;
}

void FeatureTrackerGPU::freeStorage(const std::size_t & camera_id) {
  // layer of indirection (mapped memory)
  if(buffer_[camera_id].h_indir_data_ != nullptr) {
    CUDA_API_CALL(cudaFreeHost(buffer_[camera_id].h_indir_data_));
    buffer_[camera_id].h_indir_data_ = nullptr;
  }
  // metadata (mapped memory)
  if(buffer_[camera_id].h_metadata_ != nullptr) {
    CUDA_API_CALL(cudaFreeHost(buffer_[camera_id].h_metadata_));
    buffer_[camera_id].h_metadata_ = nullptr;
  }
  // patch data
  if(buffer_[camera_id].d_patch_data_ != nullptr) {
    CUDA_API_CALL(cudaFree(buffer_[camera_id].d_patch_data_));
    buffer_[camera_id].d_patch_data_ = nullptr;
  }
  // hessian data (inverse)
  if(buffer_[camera_id].d_hessian_data_ != nullptr) {
    CUDA_API_CALL(cudaFree(buffer_[camera_id].d_hessian_data_));
    buffer_[camera_id].d_hessian_data_ = nullptr;
  }
}

void FeatureTrackerGPU::initBufferIds(const std::size_t & camera_id) {
  buffer_[camera_id].available_indices_.reserve(max_ftr_count_);
  for(std::size_t bin_id=0;bin_id<max_ftr_count_;++bin_id) {
    buffer_[camera_id].available_indices_.push_back(max_ftr_count_-bin_id-1);
  }
}

std::size_t FeatureTrackerGPU::acquireBufferId(GPUBuffer *buffer) {
  assert(buffer->available_indices_.size() > 0);
  std::size_t last_element = buffer->available_indices_.back();
  buffer->available_indices_.pop_back();
  return last_element;
}

void FeatureTrackerGPU::releaseBufferId(const std::size_t & id,
                                        const std::size_t & camera_id) {
  releaseBufferId(&buffer_[camera_id], id);
}

void FeatureTrackerGPU::reset(void) {
  // release buffer ids
  for(std::size_t c=0;c<tracks_.size();++c) {
    for(const struct FeatureTrack & track : tracks_[c]) {
      releaseBufferId(track.buffer_id_,c);
    }
  }
  // clear the parent member variables
  FeatureTrackerBase::reset();
}

} // namespace vilib
