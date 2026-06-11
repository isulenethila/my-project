"use client";

import { useState, useRef, useEffect } from "react";
import axios from "axios";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const CLASS_COLORS: Record<string, string> = {
  No_DR: "bg-green-100 text-green-800 border-green-300",
  Mild: "bg-yellow-100 text-yellow-800 border-yellow-300",
  Moderate: "bg-orange-100 text-orange-800 border-orange-300",
  Severe: "bg-red-100 text-red-800 border-red-300",
  Proliferate_DR: "bg-purple-100 text-purple-800 border-purple-300",
};

const CLASS_BAR_COLORS: Record<string, string> = {
  No_DR: "bg-green-500",
  Mild: "bg-yellow-500",
  Moderate: "bg-orange-500",
  Severe: "bg-red-500",
  Proliferate_DR: "bg-purple-500",
};

interface PredictionResult {
  prediction: string;
  confidence: number;
  probabilities: Record<string, number>;
  recommendation: string;
  original_image: string;
  clahe_image: string;
  gradcam_image: string;
}

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [patientName, setPatientName] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetch(`${API_URL}/health`).catch(() => {});
  }, []);
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setSelectedFile(file);
    setResult(null);
    setError(null);
    const reader = new FileReader();
    reader.onload = (ev) => setPreview(ev.target?.result as string);
    reader.readAsDataURL(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (!file) return;
    if (!file.type.includes("image")) {
      setError("Please upload a JPG or PNG image.");
      return;
    }
    setSelectedFile(file);
    setResult(null);
    setError(null);
    const reader = new FileReader();
    reader.onload = (ev) => setPreview(ev.target?.result as string);
    reader.readAsDataURL(file);
  };

  const handleSubmit = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append("file", selectedFile);
    try {
      const response = await axios.post<PredictionResult>(
        `${API_URL}/predict`,
        formData,
        { headers: { "Content-Type": "multipart/form-data" } },
      );
      setResult(response.data);
    } catch (err: any) {
      setError(
        err.response?.data?.detail ||
          "Analysis failed. Please check the backend is running and try again.",
      );
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadPDF = async () => {
    if (!result) return;
    const formData = new FormData();
    formData.append("patient_name", patientName);
    formData.append("prediction", result.prediction);
    formData.append("confidence", result.confidence.toString());
    formData.append("probabilities", JSON.stringify(result.probabilities));
    formData.append("original_b64", result.original_image);
    formData.append("gradcam_b64", result.gradcam_image);
    try {
      const response = await axios.post(`${API_URL}/report`, formData, {
        responseType: "blob",
      });
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", `DR_Report_${result.prediction}.pdf`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch {
      setError("Failed to generate PDF. Please try again.");
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setPatientName("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <main className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm border-b sticky top-0 z-10">
        <div className="max-w-5xl mx-auto px-4 py-4 flex items-center gap-3">
          <span className="text-3xl">👁️</span>
          <div>
            <h1 className="text-xl font-bold text-gray-800">
              DR Detection System
            </h1>
            <p className="text-xs text-gray-500">
              Explainable Deep Learning for Diabetic Retinopathy &bull;
              University of Vavuniya &bull; K.W.I.N. Kariyawasam
            </p>
          </div>
        </div>
      </header>

      <div className="max-w-5xl mx-auto px-4 py-8 space-y-6">
        <div className="bg-white rounded-2xl shadow-sm border p-6">
          <h2 className="text-base font-semibold text-gray-700 mb-4">
            Step 1 — Enter Patient Details
          </h2>
          <input
            type="text"
            placeholder="Patient Name (optional)"
            value={patientName}
            onChange={(e) => setPatientName(e.target.value)}
            className="w-full border border-gray-200 rounded-xl px-4 py-2.5
                       text-sm mb-5 focus:outline-none focus:ring-2
                       focus:ring-blue-300 focus:border-transparent"
          />
          <h2 className="text-base font-semibold text-gray-700 mb-3">
            Step 2 — Upload Retinal Fundus Image
          </h2>
          <div
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-2xl p-8 text-center
                        cursor-pointer transition-all
                        ${
                          preview
                            ? "border-blue-300 bg-blue-50"
                            : "border-gray-200 hover:border-blue-300 hover:bg-blue-50"
                        }`}
          >
            {preview ? (
              <div>
                <img
                  src={preview}
                  alt="Preview"
                  className="max-h-52 mx-auto rounded-xl object-contain mb-2"
                />
                <p className="text-xs text-blue-600">Click to change image</p>
              </div>
            ) : (
              <div>
                <p className="text-4xl mb-3">🔬</p>
                <p className="text-gray-600 font-medium">
                  Drop retinal fundus image here
                </p>
                <p className="text-gray-400 text-sm mt-1">
                  or click to browse — supports JPG, PNG
                </p>
              </div>
            )}
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/jpg,image/png"
            className="hidden"
            onChange={handleFileChange}
          />
          <div className="flex gap-3 mt-4">
            <button
              onClick={handleSubmit}
              disabled={!selectedFile || loading}
              className="flex-1 bg-blue-600 hover:bg-blue-700
                         disabled:bg-gray-200 disabled:cursor-not-allowed
                         text-white disabled:text-gray-400
                         font-semibold py-3 rounded-xl transition-colors text-sm"
            >
              {loading ? "⏳ Analysing image..." : "🔍 Analyse Image"}
            </button>
            {(selectedFile || result) && (
              <button
                onClick={handleReset}
                className="px-6 bg-gray-100 hover:bg-gray-200 text-gray-600
                           font-semibold py-3 rounded-xl transition-colors text-sm"
              >
                ↺ Reset
              </button>
            )}
          </div>
          {error && (
            <div className="mt-3 bg-red-50 border border-red-200 rounded-xl p-3 text-sm text-red-700">
              ⚠️ {error}
            </div>
          )}
        </div>

        {result && (
          <>
            <div className="bg-white rounded-2xl shadow-sm border p-6">
              <h2 className="text-base font-semibold text-gray-700 mb-4">
                Prediction Result
              </h2>
              <div className="flex flex-wrap items-center gap-4 mb-5">
                <div
                  className={`px-6 py-3 rounded-2xl border-2 text-lg font-bold ${CLASS_COLORS[result.prediction]}`}
                >
                  {result.prediction.replace(/_/g, " ")}
                </div>
                <div>
                  <p className="text-3xl font-bold text-gray-800">
                    {result.confidence.toFixed(1)}%
                  </p>
                  <p className="text-xs text-gray-400 uppercase tracking-wide">
                    Confidence
                  </p>
                </div>
              </div>
              <div className="bg-blue-50 border border-blue-100 rounded-xl p-4 text-sm text-blue-800 mb-5">
                <span className="font-semibold">💡 Recommendation: </span>
                {result.recommendation}
              </div>
              <p className="text-sm font-medium text-gray-500 mb-3">
                All Class Probabilities:
              </p>
              <div className="space-y-2.5">
                {Object.entries(result.probabilities).map(([cls, prob]) => (
                  <div key={cls} className="flex items-center gap-3">
                    <span className="w-36 text-xs text-gray-600 text-right shrink-0">
                      {cls.replace(/_/g, " ")}
                    </span>
                    <div className="flex-1 bg-gray-100 rounded-full h-4">
                      <div
                        className={`h-4 rounded-full transition-all duration-700 ${CLASS_BAR_COLORS[cls]}`}
                        style={{ width: `${prob}%` }}
                      />
                    </div>
                    <span className="w-14 text-xs font-semibold text-gray-700 shrink-0">
                      {prob.toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-white rounded-2xl shadow-sm border p-6">
              <h2 className="text-base font-semibold text-gray-700 mb-1">
                Retinal Image Analysis
              </h2>
              <p className="text-xs text-gray-400 mb-5">
                Score-CAM (Wang et al. 2020) highlights retinal regions that
                influenced the prediction — gradient-free XAI. Red/yellow = high
                attention &bull; Blue = low attention
              </p>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <img
                    src={result.original_image}
                    alt="Original"
                    className="w-full rounded-xl border object-cover aspect-square"
                  />
                  <p className="text-xs text-center text-gray-500 mt-2 font-medium">
                    Original Image
                  </p>
                </div>
                <div>
                  <img
                    src={result.clahe_image}
                    alt="CLAHE Enhanced"
                    className="w-full rounded-xl border object-cover aspect-square"
                  />
                  <p className="text-xs text-center text-gray-500 mt-2 font-medium">
                    CLAHE Enhanced{" "}
                    <span className="text-gray-400">(Model Input)</span>
                  </p>
                </div>
                <div>
                  <img
                    src={result.gradcam_image}
                    alt="Grad-CAM"
                    className="w-full rounded-xl border object-cover aspect-square"
                  />
                  <p className="text-xs text-center text-gray-500 mt-2 font-medium">
                    Score-CAM Heatmap{" "}
                    <span className="text-red-400">🔴 Focus areas</span>
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-2xl shadow-sm border p-6">
              <h2 className="text-base font-semibold text-gray-700 mb-1">
                Download Clinical Report
              </h2>
              <p className="text-sm text-gray-400 mb-4">
                Generate a PDF with prediction, Grad-CAM heatmap, probabilities
                and recommendation.
              </p>
              <button
                onClick={handleDownloadPDF}
                className="bg-green-600 hover:bg-green-700 text-white font-semibold py-3 px-8 rounded-xl transition-colors text-sm"
              >
                📄 Download PDF Report
              </button>
            </div>
          </>
        )}

        <footer className="text-center text-xs text-gray-400 pb-8 space-y-1">
          <p>
            K.W.I.N. Kariyawasam | Reg. No: 2020/ICT/39 | University of
            Vavuniya, Department of Physical Science
          </p>
          <p>
            ⚠️ This system is a research prototype. It should not replace
            clinical diagnosis by a qualified ophthalmologist.
          </p>
        </footer>
      </div>
    </main>
  );
}
