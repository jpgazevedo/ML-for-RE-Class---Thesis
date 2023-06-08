import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import { Routes, Route, BrowserRouter } from 'react-router-dom';
import ManualImport from './pages/ManualImport';
import Configs from './pages/Configs';
import TrainBatchLabel from './pages/TrainBatchLabel';
import DownloadFile from './pages/DownloadFile';
import ImportMethod from './pages/ImportMethod';
import FileImport from './pages/FileImport';
import TestSetLabeling from './pages/TestSetLabeling';
import MetricsOutput from './pages/MetricsOutput';


const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <BrowserRouter>
        <Routes>
          <Route index element={<App/>}/>
          <Route path="/configs" element={<Configs/>}/>
          <Route path="/configs/req-inputConfigs" element={<ImportMethod/>}/>
          <Route path="/configs/req-inputConfigs/manualInput" element={<ManualImport/>}/>
          <Route path="/configs/req-inputConfigs/fileInput" element={<FileImport/>}/>
          <Route path="/configs/req-input/test-set-labeling" element={<TestSetLabeling/>}/>
          <Route path="/configs/req-input/test-set-labeling/train-batch-labeling" element={<TrainBatchLabel/>}/>
          <Route path="/configs/req-input/test-set-labeling/train-batch-labeling/stopFlg" element={<MetricsOutput/>}/>
          <Route path="/configs/req-input/test-set-labeling/train-batch-labeling/stopFlg/downloadFile" element={<DownloadFile/>}/>
        </Routes>
      </BrowserRouter>
    </React.StrictMode>
  
);

