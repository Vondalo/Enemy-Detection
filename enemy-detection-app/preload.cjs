const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  selectImage: () => ipcRenderer.invoke('select-image'),
  runPrediction: (imagePath) => ipcRenderer.invoke('run-prediction', imagePath),
  runPipelineStep: (stepName, args) => ipcRenderer.invoke('run-pipeline-step', stepName, args),
  onPipelineOutput: (callback) => ipcRenderer.on('pipeline-output', (_event, data) => callback(data)),
  removePipelineOutputListener: () => ipcRenderer.removeAllListeners('pipeline-output'),
  saveLinks: (linksText) => ipcRenderer.invoke('save-links', linksText),
  clearVideos: () => ipcRenderer.invoke('clear-videos')
});
