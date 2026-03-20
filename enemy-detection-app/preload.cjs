const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  selectImage: () => ipcRenderer.invoke('select-image'),
  runPrediction: (imagePath) => ipcRenderer.invoke('run-prediction', imagePath),
  runPipelineStep: (stepName, args) => ipcRenderer.invoke('run-pipeline-step', stepName, args),
  onPipelineOutput: (callback) => ipcRenderer.on('pipeline-output', (_event, data) => callback(data)),
  removePipelineOutputListener: () => ipcRenderer.removeAllListeners('pipeline-output'),
  saveLinks: (linksText) => ipcRenderer.invoke('save-links', linksText),
  clearVideos: () => ipcRenderer.invoke('clear-videos'),
  cancelPipeline: () => ipcRenderer.invoke('cancel-pipeline'),
  
  // New Dataset Management APIs
  listDatasets: () => ipcRenderer.invoke('list-datasets'),
  analyzeDatasetBias: (datasetPath, csvName) => ipcRenderer.invoke('analyze-dataset-bias', datasetPath, csvName),
  runBiasFix: (datasetPath, csvName) => ipcRenderer.invoke('run-bias-fix', datasetPath, csvName),
  runTraining: (datasetPath, csvName, epochs) => ipcRenderer.invoke('run-training', datasetPath, csvName, epochs),
  listVideos: () => ipcRenderer.invoke('list-videos'),
  runDataCollection: (videoName, datasetName) => ipcRenderer.invoke('run-data-collection', videoName, datasetName),
  runAugmentation: (datasetPath, csvName, outputDatasetName) => ipcRenderer.invoke('run-augmentation', datasetPath, csvName, outputDatasetName),
  
  // Window Controls
  minimizeWindow: () => ipcRenderer.send('window-minimize'),
  maximizeWindow: () => ipcRenderer.send('window-maximize'),
  closeWindow: () => ipcRenderer.send('window-close')
});
