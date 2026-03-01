export interface ModelData {
  id: string
  name: string
  fullName: string
  accuracy: number
  topFiveAccuracy: number
  parameters: string
  parametersNum: number
  inferenceTime: number
  pros: string[]
  cons: string[]
  verdict: "winner" | "eliminated"
  eliminationReason?: string
}

export const MODELS: ModelData[] = [
  {
    id: "resnet50",
    name: "ResNet-50",
    fullName: "Residual Network 50",
    accuracy: 91.2,
    topFiveAccuracy: 96.8,
    parameters: "25.6M",
    parametersNum: 25.6,
    inferenceTime: 45,
    pros: ["Well-established", "Good accuracy"],
    cons: ["Heavy model", "Slower inference"],
    verdict: "eliminated",
    eliminationReason: "Too heavy for real-time use",
  },
  {
    id: "mobilenetv3",
    name: "MobileNetV3",
    fullName: "MobileNet Version 3",
    accuracy: 88.7,
    topFiveAccuracy: 94.2,
    parameters: "5.4M",
    parametersNum: 5.4,
    inferenceTime: 12,
    pros: ["Lightweight", "Fast inference"],
    cons: ["Lower accuracy", "Less robust"],
    verdict: "eliminated",
    eliminationReason: "Accuracy below threshold",
  },
  {
    id: "vgg16",
    name: "VGG-16",
    fullName: "Visual Geometry Group 16",
    accuracy: 89.3,
    topFiveAccuracy: 95.1,
    parameters: "138M",
    parametersNum: 138,
    inferenceTime: 78,
    pros: ["Simple architecture"],
    cons: ["Extremely heavy", "Very slow"],
    verdict: "eliminated",
    eliminationReason: "Impractical model size",
  },
  {
    id: "densenet121",
    name: "DenseNet-121",
    fullName: "Densely Connected Network 121",
    accuracy: 93.1,
    topFiveAccuracy: 97.2,
    parameters: "8M",
    parametersNum: 8,
    inferenceTime: 38,
    pros: ["Good accuracy", "Efficient parameters"],
    cons: ["Memory intensive", "Moderate speed"],
    verdict: "eliminated",
    eliminationReason: "Good but not optimal",
  },
  {
    id: "efficientnetb0",
    name: "EfficientNet-B0",
    fullName: "EfficientNet Baseline",
    accuracy: 94.2,
    topFiveAccuracy: 97.8,
    parameters: "5.3M",
    parametersNum: 5.3,
    inferenceTime: 18,
    pros: ["Efficient", "Good balance"],
    cons: ["B2 performs better"],
    verdict: "eliminated",
    eliminationReason: "B2 variant superior",
  },
  {
    id: "efficientnetb2",
    name: "EfficientNet-B2",
    fullName: "EfficientNet B2 (Noisy Student)",
    accuracy: 96.06,
    topFiveAccuracy: 98.74,
    parameters: "9.2M",
    parametersNum: 9.2,
    inferenceTime: 24,
    pros: ["Best accuracy", "Optimal efficiency", "Robust to noise"],
    cons: [],
    verdict: "winner",
  },
]

export const SCENE_DURATIONS = [3, 5, 10, 4, 5] // seconds
export const TOTAL_DURATION = SCENE_DURATIONS.reduce((a, b) => a + b, 0)
