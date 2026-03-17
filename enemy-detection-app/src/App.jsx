import { useState, useRef, useEffect } from 'react'
import './index.css'

function App() {
  const [imagePath, setImagePath] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [truth, setTruth] = useState(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("Ready");
  
  const imgRef = useRef(null);
  const [imgDims, setImgDims] = useState({ width: 0, height: 0 });

  const handleSelectImage = async () => {
    const path = await window.electronAPI.selectImage();
    if (path) {
      setImagePath(path);
      setPrediction(null);
      setTruth(null);
      setStatus("Image Loaded");
    }
  };

  const handlePredict = async () => {
    if (!imagePath) return;
    
    setLoading(true);
    setStatus("Analyzing...");
    
    try {
      const result = await window.electronAPI.runPrediction(imagePath);
      if (result.error) {
        setStatus(`Error: ${result.error}`);
      } else {
        setPrediction(result.prediction);
        setTruth(result.truth);
        setStatus("Analysis Complete");
      }
    } catch (err) {
      setStatus(`System Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const updateImageDims = () => {
    if (imgRef.current) {
      setImgDims({
        width: imgRef.current.clientWidth,
        height: imgRef.current.clientHeight
      });
    }
  };

  useEffect(() => {
    window.addEventListener('resize', updateImageDims);
    return () => window.removeEventListener('resize', updateImageDims);
  }, []);

  return (
    <div className="app-container">
      <header className="header">
        <h1>Enemy Localization Tester</h1>
        <div className="controls">
          <button onClick={handleSelectImage} disabled={loading}>
            Select Image
          </button>
          <button 
            className="primary" 
            onClick={handlePredict} 
            disabled={!imagePath || loading}
          >
            {loading ? "Detecting..." : "Detect Enemy"}
          </button>
        </div>
      </header>

      <div className="viewer-content">
        {!imagePath ? (
          <div className="empty-state">
            <p>Please select a screenshot to begin analysis</p>
          </div>
        ) : (
          <div className="image-wrapper">
            <img 
              ref={imgRef}
              src={`file://${imagePath}`} 
              alt="Enemy Screenshot" 
              onLoad={updateImageDims}
            />
            
            {prediction && (
              <div 
                className="coordinate-dot dot-predicted"
                style={{
                  left: `${prediction[0] * imgDims.width}px`,
                  top: `${prediction[1] * imgDims.height}px`
                }}
                title={`Predicted: ${prediction[0].toFixed(3)}, ${prediction[1].toFixed(3)}`}
              />
            )}

            {truth && (
              <div 
                className="coordinate-dot dot-truth"
                style={{
                  left: `${truth[0] * imgDims.width}px`,
                  top: `${truth[1] * imgDims.height}px`
                }}
                title={`Ground Truth: ${truth[0].toFixed(3)}, ${truth[1].toFixed(3)}`}
              />
            )}

            {(prediction || truth) && (
              <div className="legend">
                <div className="legend-item">
                  <div className="dot-mini dot-predicted"></div>
                  <span>AI Prediction</span>
                </div>
                {truth && (
                  <div className="legend-item">
                    <div className="dot-mini dot-truth"></div>
                    <span>Ground Truth</span>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      <footer className="status-bar">
        Status: <span>{status}</span>
        {imagePath && (
          <span style={{ marginLeft: '1rem' }}>
            File: <span className="pixel-coords">{imagePath}</span>
          </span>
        )}
      </footer>
    </div>
  )
}

export default App
