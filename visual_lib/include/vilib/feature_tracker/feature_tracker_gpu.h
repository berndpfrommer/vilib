/*
 * Feature tracker GPU
 * feature_tracker_gpu.h
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

#pragma once

#include <vector>
#include "vilib/feature_tracker/feature_tracker_base.h"
#include "vilib/feature_detection/detector_base_gpu.h"
#include "vilib/common/feature.h"
#include "vilib/common/pyramid_info.h"

namespace vilib {

class FeatureTrackerGPU : public FeatureTrackerBase {
public:
  FeatureTrackerGPU(const FeatureTrackerOptions & options,
                    const std::size_t & camera_num);
  ~FeatureTrackerGPU(void);

  void track(const std::shared_ptr<FrameBundle> & cur_frames,
             std::size_t & total_tracked_features_num,
             std::size_t & total_detected_features_num) override;
  void setDetectorGPU(std::shared_ptr<DetectorBaseGPU> & detector,
                      const std::size_t & camera_id) override;
  void reset(void) override;
  void trackFeatures(std::vector<PyramidInfo> *prev_pyr,
                     std::vector<PyramidInfo> *cur_pyr,
                     const std::vector<std::vector<Feature>> &prev_features);
  void computePyramidInfo(std::vector<PyramidInfo> *pyr,
                          const std::shared_ptr<FrameBundle> & frames);

private:
  struct GPUBuffer {
    // Indirection layer
    int * h_indir_data_;
    int * d_indir_data_;
    std::vector<std::size_t> available_indices_;
    // Metadata
    unsigned char * h_metadata_;
    unsigned char * d_metadata_;
    float4 * d_cur_f_;
    float  * h_cur_f_;
    float2 * d_template_px_;
    float  * h_template_px_;
    float2 * d_first_px_;
    float  * h_first_px_;
    float2 * d_cur_px_;
    float  * h_cur_px_;
    float2 * d_cur_alpha_beta_;
    float  * h_cur_alpha_beta_;
    float  * d_cur_disparity_;
    float  * h_cur_disparity_;
    // Patch data
    unsigned char * d_patch_data_;
    // Hessian data (inverse)
    float * d_hessian_data_;
  };

  void freeStorage(const::std::size_t & camera_id);
  void clearTracksAndFeatures(
    const std::shared_ptr<Frame> &base_frame,
    GPUBuffer *buffer_ptr,
    std::vector<FeatureTrack> *tracks, size_t *tracked_features_num);

  void trackOnGPU(const image_pyramid_descriptor_t & pyramid_description,
                  const pyramid_patch_descriptor_t & pyramid_patch_sizes,
                  const std::vector<FeatureTrack> &tracks,
                  struct GPUBuffer *buffer_ptr,
                  cudaStream_t *stream_ptr);

  void processTrackingResults(
    const std::shared_ptr<Frame> &base_frame,
    std::vector<FeatureTrack> *tracks,
    size_t *tracked_features_num,
    struct GPUBuffer *buffer_ptr,
    cudaStream_t *stream_ptr);

  void addFeaturesToTracks(
    const std::shared_ptr<Frame> &base_frame,
    const image_pyramid_descriptor_t &pyramids,
    const std::vector<Feature> &features,
    size_t tracked_features_num,
    GPUBuffer *buffer,
    size_t *detected_features_num,
    std::vector<FeatureTrack> *tracks,
    cudaStream_t *stream_ptr);

  void detectNewFeatures(std::vector<Feature> *features,
                         const Frame &base_frame,
                         size_t max_new_features,
                         const std::shared_ptr<DetectorBaseGPU> &detector);

  int addTrack(GPUBuffer *buffer_ptr,
               std::vector<FeatureTrack> *tracks,
               const std::shared_ptr<Frame> & first_frame,
               const float & first_x,
               const float & first_y,
               const int & first_level,
               const float & first_score);
  void updateTracks(const std::size_t & last_n,
                    const image_pyramid_descriptor_t & pyramid_description,
                    GPUBuffer *buffer,
                    std::vector<FeatureTrack> *tracks,
                    cudaStream_t *stream_ptr);
  // Buffer management (indirection layer)
  void initBufferIds(const std::size_t & camera_id);
  std::size_t acquireBufferId(GPUBuffer *buffer);
  inline void releaseBufferId(GPUBuffer *buffer,
                              const std::size_t & id) {
    buffer->available_indices_.push_back(id);
  }
  void releaseBufferId(const std::size_t & id,
                       const std::size_t & camera_id);

  std::size_t pyramid_levels_;
  pyramid_patch_descriptor_t pyramid_patch_sizes_;
  std::vector<struct GPUBuffer> buffer_;
  std::vector<cudaStream_t> stream_;
  std::vector<std::shared_ptr<DetectorBaseGPU>> detector_;
  std::size_t max_ftr_count_;
};

} // namespace vilib
