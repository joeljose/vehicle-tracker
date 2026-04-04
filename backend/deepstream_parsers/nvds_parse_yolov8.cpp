/*
 * Custom NvDsInfer parser for YOLOv8s raw output (1, 84, 8400).
 *
 * Performs confidence filtering + vehicle class filter + greedy NMS.
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

static const float CONF_THRESH = 0.25f;
static const float IOU_THRESH = 0.45f;
static const int NUM_CLASSES = 80;
static const int NUM_ANCHORS = 8400;

static const int VEHICLE_CLASSES[] = {2, 3, 5, 7};
static const int NUM_VEHICLE_CLASSES = 4;

static bool is_vehicle(int cls) {
    for (int i = 0; i < NUM_VEHICLE_CLASSES; i++)
        if (VEHICLE_CLASSES[i] == cls) return true;
    return false;
}

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
        if (layer.inferDims.numElements >= 84 * NUM_ANCHORS) {
            data = static_cast<const float*>(layer.buffer);
            break;
        }
    }
    if (!data) return false;

    // data layout: [84][8400] row-major
    struct Det { float x1, y1, w, h, score; int cls; };
    std::vector<Det> dets;

    for (int a = 0; a < NUM_ANCHORS; a++) {
        float best_score = 0;
        int best_cls = 0;
        for (int c = 0; c < NUM_CLASSES; c++) {
            float s = data[(4 + c) * NUM_ANCHORS + a];
            if (s > best_score) { best_score = s; best_cls = c; }
        }
        if (best_score < CONF_THRESH || !is_vehicle(best_cls)) continue;

        float cx = data[0 * NUM_ANCHORS + a];
        float cy = data[1 * NUM_ANCHORS + a];
        float w  = data[2 * NUM_ANCHORS + a];
        float h  = data[3 * NUM_ANCHORS + a];

        Det d;
        d.x1 = cx - w / 2.0f;
        d.y1 = cy - h / 2.0f;
        d.w = w;
        d.h = h;
        d.score = best_score;
        d.cls = best_cls;
        dets.push_back(d);
    }

    std::sort(dets.begin(), dets.end(),
        [](const Det& a, const Det& b) { return a.score > b.score; });

    // Greedy NMS
    std::vector<bool> suppressed(dets.size(), false);
    for (size_t i = 0; i < dets.size(); i++) {
        if (suppressed[i]) continue;

        NvDsInferObjectDetectionInfo obj;
        obj.classId = dets[i].cls;
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
