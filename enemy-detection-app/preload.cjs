const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  selectImage: () => ipcRenderer.invoke('select-image'),
  selectVideo: () => ipcRenderer.invoke('select-video'),
  runPrediction: (imagePath) => ipcRenderer.invoke('run-prediction', imagePath),
  startVideoPrediction: (payload) => ipcRenderer.invoke('start-video-prediction', payload),
  stopVideoPrediction: () => ipcRenderer.invoke('stop-video-prediction'),
  runPipelineStep: (stepName, args) => ipcRenderer.invoke('run-pipeline-step', stepName, args),
  onPipelineOutput: (callback) => ipcRenderer.on('pipeline-output', (_event, data) => callback(data)),
  onVideoPredictionEvent: (callback) => ipcRenderer.on('video-prediction-event', (_event, data) => callback(data)),
  removePipelineOutputListener: () => ipcRenderer.removeAllListeners('pipeline-output'),
  removeVideoPredictionListener: () => ipcRenderer.removeAllListeners('video-prediction-event'),
  saveLinks: (linksText) => ipcRenderer.invoke('save-links', linksText),
  clearVideos: () => ipcRenderer.invoke('clear-videos'),
  cancelPipeline: () => ipcRenderer.invoke('cancel-pipeline'),
  
  // New Dataset Management APIs
  listDatasets: () => ipcRenderer.invoke('list-datasets'),
  mergeDatasets: (payload) => ipcRenderer.invoke('merge-datasets', payload),
  listDatasetImages: (datasetPath, csvName) => ipcRenderer.invoke('list-dataset-images', datasetPath, csvName),
  saveDatasetImageAnnotations: (payload) => ipcRenderer.invoke('save-dataset-image-annotations', payload),
  deleteDatasetImage: (datasetPath, csvName, filename) => ipcRenderer.invoke('delete-dataset-image', datasetPath, csvName, filename),
  analyzeDatasetBias: (datasetPath, csvName) => ipcRenderer.invoke('analyze-dataset-bias', datasetPath, csvName),
  runBiasFix: (datasetPath, csvName) => ipcRenderer.invoke('run-bias-fix', datasetPath, csvName),
  runTraining: (payload) => ipcRenderer.invoke('run-training', payload),
  listTrainingRuns: () => ipcRenderer.invoke('list-training-runs'),
  loadTrainingRun: (summaryPath) => ipcRenderer.invoke('load-training-run', summaryPath),
  listVideos: () => ipcRenderer.invoke('list-videos'),
  runDataCollection: (videoName, datasetName) => ipcRenderer.invoke('run-data-collection', videoName, datasetName),
  startManualCollection: (videoName, datasetName) => ipcRenderer.invoke('start-manual-collection', videoName, datasetName),
  saveManualAnnotation: (payload) => ipcRenderer.invoke('save-manual-annotation', payload),
  finishManualCollection: (datasetPath) => ipcRenderer.invoke('finish-manual-collection', datasetPath),
  runAugmentation: (datasetPath, csvName, outputDatasetName) => ipcRenderer.invoke('run-augmentation', datasetPath, csvName, outputDatasetName),
  
  // Window Controls
  minimizeWindow: () => ipcRenderer.send('window-minimize'),
  maximizeWindow: () => ipcRenderer.send('window-maximize'),
  closeWindow: () => ipcRenderer.send('window-close')
});
