/*
 * Custom NvDsInfer parser for YOLOv8 raw output.
 *
 * M8-P1.5 v2: the model is a single-class fine-tuned YOLOv8s. Output
 * tensor shape is (1, 5, 8400) — 4 bbox coords + 1 class score per anchor.
 * All detections are class 0 (vehicle) by construction.
 *
 * Under the original COCO-pretrained yolov8s the output was (1, 84, 8400)
 * with 80 class logits, and this parser also ran a COCO vehicle-class
 * filter (ids 2/3/5/7). That filter is no longer needed because there
 * is only one class.
 *
 * Compiled with: c++ -shared -fPIC -O2 -std=c++17 -I<DS>/sources/includes ...
 */

#include <cmath>
#include <algorithm>
#include <vector>

// Include nvdsinfer.h for core types (no CUDA dependency)
#include "nvdsinfer.h"

// From nvdsinfer_custom_impl.h — inlined to avoid CUDA header chain
typedef struct {
    unsigned int numClassesConfigured;
    std::vector<float> perClassPreclusterThreshold;
    std::vector<float> perClassPostclusterThreshold;
    std::vector<float> &perClassThreshold = perClassPreclusterThreshold;
} NvDsInferParseDetectionParams;

typedef bool (* NvDsInferParseCustomFunc) (
        std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsInferParseDetectionParams const &detectionParams,
        std::vector<NvDsInferObjectDetectionInfo> &objectList);

#define CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(customParseFunc) \
    static NvDsInferParseCustomFunc const __check ## customParseFunc \
        __attribute__((unused)) = customParseFunc;

// Post-processing thresholds. Slightly lower conf than the 0.25 default
// because the single-class student's confidence distribution is calibrated
// higher than the COCO-pretrained model (mean 0.80 vs 0.59 at inference),
// but many true positives still sit in the 0.25-0.4 band.
static const float CONF_THRESH = 0.25f;
static const float IOU_THRESH = 0.45f;
static const int NUM_CLASSES = 1;      // single-class model
static const int NUM_ANCHORS = 8400;
// YOLOv8 output is [4 bbox + NUM_CLASSES scores] per anchor, transposed
// to (4 + NUM_CLASSES, NUM_ANCHORS) in the network output layer.
static const int NUM_CHANNELS = 4 + NUM_CLASSES;  // = 5

static float compute_iou(float ax1, float ay1, float aw, float ah,
                         float bx1, float by1, float bw, float bh) {
    float x1 = std::max(ax1, bx1);
    float y1 = std::max(ay1, by1);
    float x2 = std::min(ax1 + aw, bx1 + bw);
    float y2 = std::min(ay1 + ah, by1 + bh);
    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float area_a = aw * ah;
    float area_b = bw * bh;
    return inter / (area_a + area_b - inter + 1e-6f);
}

extern "C"
bool NvDsInferParseYoloV8(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList)
{
    if (outputLayersInfo.empty()) return false;

    const float* data = nullptr;
    for (const auto& layer : outputLayersInfo) {
        if (layer.inferDims.numElements >= (unsigned int)(NUM_CHANNELS * NUM_ANCHORS)) {
            data = static_cast<const float*>(layer.buffer);
            break;
        }
    }
    if (!data) return false;

    // data layout: [NUM_CHANNELS][NUM_ANCHORS] row-major.
    // With NUM_CHANNELS=5: rows 0-3 are cx/cy/w/h, row 4 is the single
    // "vehicle" class score.
    struct Det { float x1, y1, w, h, score; };
    std::vector<Det> dets;
    dets.reserve(256);

    for (int a = 0; a < NUM_ANCHORS; a++) {
        // Single-class: the score is just data[4][a]. No argmax needed.
        float score = data[4 * NUM_ANCHORS + a];
        if (score < CONF_THRESH) continue;

        float cx = data[0 * NUM_ANCHORS + a];
        float cy = data[1 * NUM_ANCHORS + a];
        float w  = data[2 * NUM_ANCHORS + a];
        float h  = data[3 * NUM_ANCHORS + a];

        Det d;
        d.x1 = cx - w / 2.0f;
        d.y1 = cy - h / 2.0f;
        d.w = w;
        d.h = h;
        d.score = score;
        dets.push_back(d);
    }

    std::sort(dets.begin(), dets.end(),
        [](const Det& a, const Det& b) { return a.score > b.score; });

    // Greedy NMS. All detections are class 0 so we don't need to
    // partition by class before suppressing.
    std::vector<bool> suppressed(dets.size(), false);
    for (size_t i = 0; i < dets.size(); i++) {
        if (suppressed[i]) continue;

        NvDsInferObjectDetectionInfo obj;
        obj.classId = 0;  // always "vehicle" under the single-class model
        obj.detectionConfidence = dets[i].score;
        obj.left   = dets[i].x1;
        obj.top    = dets[i].y1;
        obj.width  = dets[i].w;
        obj.height = dets[i].h;
        objectList.push_back(obj);

        for (size_t j = i + 1; j < dets.size(); j++) {
            if (suppressed[j]) continue;
            if (compute_iou(dets[i].x1, dets[i].y1, dets[i].w, dets[i].h,
                            dets[j].x1, dets[j].y1, dets[j].w, dets[j].h) > IOU_THRESH)
                suppressed[j] = true;
        }
    }
    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloV8);
