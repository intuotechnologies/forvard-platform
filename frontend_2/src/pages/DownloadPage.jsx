// src/pages/DPage.jsx

import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import financialDataService from "../services/financialDataService";
import "../styles/download.css";
import { Download, Lock, CheckCircle, AlertTriangle } from "lucide-react";

const DownloadPage = () => {
  const [accessLimits, setAccessLimits] = useState(null);
  const [downloadStates, setDownloadStates] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [jsZipLoaded, setJsZipLoaded] = useState(false);

  // Usa l'AuthContext aggiornato
  const { 
    isAuthenticated, 
    user, 
    token, 
    getAccessLimits,
    isLoading: authLoading 
  } = useAuth();
  const navigate = useNavigate();

  // Carica JSZip dinamicamente
  useEffect(() => {
    const loadJSZip = async () => {
      try {
        if (typeof window.JSZip === 'undefined') {
          console.log('Loading JSZip library...');
          const script = document.createElement('script');
          script.src = 'https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js';
          script.onload = () => {
            console.log('JSZip loaded successfully');
            setJsZipLoaded(true);
          };
          script.onerror = () => {
            console.error('Failed to load JSZip');
            setJsZipLoaded(false);
          };
          document.head.appendChild(script);
        } else {
          setJsZipLoaded(true);
        }
      } catch (error) {
        console.error('Error loading JSZip:', error);
        setJsZipLoaded(false);
      }
    };

    loadJSZip();
  }, []);

  const assetTypes = [
    { key: 'stocks', label: 'Stocks' },
    // { key: 'etfs', label: 'ETFs' }, // TODO: Uncomment when ETFs are ready
    { key: 'forex', label: 'Exchange Rates' },
    { key: 'futures', label: 'Futures' }
  ];

  // Monitor per cambi di stato di autenticazione
  useEffect(() => {
    console.log('Auth state changed:', {
      isAuthenticated,
      userRole: user?.role_name,
      hasToken: !!token,
      authLoading
    });
  }, [isAuthenticated, user, token, authLoading]);

  // Carica i limiti di accesso dell'utente
  useEffect(() => {
    if (authLoading) {
      console.log('Auth still loading, waiting...');
      return;
    }

    if (!isAuthenticated) {
      console.log('User not authenticated, skipping access limits load');
      setLoading(false);
      return;
    }

    console.log('User authenticated, loading access limits');
    loadAccessLimits();
  }, [isAuthenticated, authLoading]);

  const loadAccessLimits = async () => {
    try {
      setError(null);
      console.log('Loading access limits for user:', user?.email, 'role:', user?.role_name);
      
      // Usa la funzione del context invece del service direttamente
      const limits = await getAccessLimits();
      
      setAccessLimits(limits);
      console.log('User access limits loaded:', limits);
      
    } catch (error) {
      console.error('Failed to load access limits:', error);
      
      // Se è un errore di autenticazione, l'AuthContext ha già fatto logout
      if (error.message.includes('Session expired') || error.message.includes('login again')) {
        console.log('Session expired, redirecting to login');
        navigate('/login');
        return;
      }
      
      // NON mostrare errore per access limits - sono opzionali
      console.log('Access limits not available, continuing without them');
    } finally {
      setLoading(false);
    }
  };

  // Gestisce il download di un dataset ZIP
  const handleDownload = async (assetType, dataType) => {
    if (!isAuthenticated) {
      navigate('/login');
      return;
    }

    const downloadKey = `${assetType}_${dataType}`;

    setDownloadStates(prev => ({
      ...prev,
      [downloadKey]: 'downloading'
    }));

    try {
      console.log('Starting ZIP download:', { assetType, dataType, userRole: user?.role_name });
      
      // Parametri espliciti per richiedere ZIP al backend
      const downloadParams = {
        asset_type: assetType,
        format: 'zip',           // Forza formato ZIP
        output_format: 'zip',    // Parametro alternativo
        archive: true,           // Richiede archivio
        zip: true,              // Flag esplicito per ZIP
        compression: 'zip',      // Tipo compressione
        multiple_files: true,    // Indica file multipli
        bundle: true,           // Raggruppa file
      };
      
      // Configura parametri specifici per tipo di dati
      if (dataType === 'variances') {
        downloadParams.data_types = [
          'rv1', 'rv5', 'rv5_ss',           // Realized Variance
          'bv1', 'bv5', 'bv_ss',           // Bipower Variation  
          'rsp1', 'rsp5', 'rsp5_ss',       // Realized Semi-Variance (+)
          'rsn1', 'rsn5', 'rsn5_ss',       // Realized Semi-Variance (-)
          'medrv1', 'medrv5', 'medrv5_ss', // Median Realized Variance
          'minrv1', 'minrv5', 'minrv5_ss', // Minimum Realized Variance
          'rq1', 'rq5', 'rq5_ss',          // Realized Quarticity
          'rk',                             // Realized Kernel
          'pv', 'gk', 'rr5'               // Range-based measures
        ];
        downloadParams.include_all_variances = true;
      } else if (dataType === 'covariances') {
          downloadParams.include_covariances = true;
          downloadParams.endpoint = '/financial-data/covariance/download';
      }
      
      console.log('ZIP download params:', downloadParams);
      
      // Passa il token esplicitamente al service
      const result = await financialDataService.downloadDataset(token, downloadParams);
      
      console.log('Download response:', result);
      
      setDownloadStates(prev => ({
        ...prev,
        [downloadKey]: 'completed'
      }));

      // Se il backend non restituisce ZIP, crealo lato client
      if (result.download_url) {
        const fullDownloadUrl = result.download_url.startsWith('http') 
          ? result.download_url 
          : `http://volare.unime.it:8443${result.download_url}`;
        
        console.log('Downloading file with authentication:', fullDownloadUrl);
        
        // Determina il nome del file dalla risposta o dai parametri
        const downloadFileName = result.file_name || 
                               result.filename || 
                               `${assetType}_${dataType}_${Date.now()}`;
        
        await downloadAndProcessFile(fullDownloadUrl, downloadFileName, assetType, dataType);
        
      } else if (result.file_content || result.data) {
        // Se il backend restituisce direttamente i dati, creali come ZIP
        console.log('Creating ZIP from response data');
        await createClientSideZip(result, `${assetType}_${dataType}.zip`, assetType, dataType);
      } else {
        throw new Error('No download URL or data received from server');
      }

      // Reset dello stato dopo 3 secondi
      setTimeout(() => {
        setDownloadStates(prev => ({
          ...prev,
          [downloadKey]: null
        }));
      }, 3000);

    } catch (error) {
      console.error('ZIP download failed:', error);
      
      let errorMessage = 'Download failed';
      
      if (error.message.includes('403')) {
        errorMessage = 'Access denied - upgrade your plan for this data';
      } else if (error.message.includes('401')) {
        errorMessage = 'Session expired - please login again';
      } else if (error.message.includes('404')) {
        errorMessage = 'Data not found';
      } else {
        errorMessage = error.message;
      }
      
      alert(`Download failed: ${errorMessage}`);
      
      setDownloadStates(prev => ({
        ...prev,
        [downloadKey]: 'error'
      }));

      // Reset dello stato dopo 3 secondi
      setTimeout(() => {
        setDownloadStates(prev => ({
          ...prev,
          [downloadKey]: null
        }));
      }, 3000);
    }
  };

  // Funzione per scaricare e processare il file (creando ZIP se necessario)
  const downloadAndProcessFile = async (downloadUrl, fileName, assetType, dataType) => {
    try {
      console.log('Downloading file from:', downloadUrl);
      
      const fileResponse = await fetch(downloadUrl, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Accept': '*/*',
        },
      });

      if (!fileResponse.ok) {
        throw new Error(`Failed to download file: ${fileResponse.status}`);
      }

      const arrayBuffer = await fileResponse.arrayBuffer();
      const uint8Array = new Uint8Array(arrayBuffer);
      
      // Verifica se è già un file ZIP
      const isZipFile = uint8Array.length >= 4 && 
                       uint8Array[0] === 0x50 && // 'P'
                       uint8Array[1] === 0x4B && // 'K' 
                       (uint8Array[2] === 0x03 || uint8Array[2] === 0x05 || uint8Array[2] === 0x07);
      
      console.log('File analysis:', {
        fileName,
        size: arrayBuffer.byteLength,
        isZip: isZipFile,
        firstBytes: Array.from(uint8Array.slice(0, 4)).map(b => `0x${b.toString(16).padStart(2, '0')}`).join(' ')
      });
      
      if (isZipFile) {
        // È già un ZIP, scaricalo direttamente
        console.log('File is already ZIP, downloading directly');
        const blob = new Blob([arrayBuffer], { type: 'application/zip' });
        const finalFileName = fileName.endsWith('.zip') ? fileName : `${fileName}.zip`;
        downloadBlob(blob, finalFileName);
      } else {
        // Non è un ZIP, probabilmente CSV - crealo come ZIP
        console.log('File is not ZIP, creating ZIP archive');
        const textContent = new TextDecoder('utf-8').decode(arrayBuffer);
        await createZipFromCSV(textContent, fileName, assetType, dataType);
      }
      
    } catch (error) {
      console.error('Download and process error:', error);
      throw error;
    }
  };

  // Funzione per creare ZIP da contenuto CSV
  const createZipFromCSV = async (csvContent, originalFileName, assetType, dataType) => {
    try {
      console.log('Creating ZIP from CSV content');
      
      // Usa JSZip se disponibile, altrimenti crea un semplice archivio
      if (jsZipLoaded && typeof window.JSZip !== 'undefined') {
        // Usa JSZip per creare un vero ZIP
        const zip = new window.JSZip();
        
        // Determina il nome del file CSV interno
        const csvFileName = originalFileName.replace('.zip', '.csv');
        zip.file(csvFileName, csvContent);
        
        // Aggiungi un file README
        const readmeContent = `Dataset: ${assetType} ${dataType}\nDownloaded: ${new Date().toISOString()}\nFormat: CSV\nSource: VOLARE Database\n`;
        zip.file('README.txt', readmeContent);
        
        // Genera il ZIP
        const zipBlob = await zip.generateAsync({ 
          type: 'blob',
          compression: 'DEFLATE',
          compressionOptions: { level: 6 }
        });
        const zipFileName = `${assetType}_${dataType}.zip`;
        
        downloadBlob(zipBlob, zipFileName);
        console.log('Real ZIP created with JSZip');
        
      } else {
        // Fallback: crea un "pseudo-ZIP" rinominando il CSV
        console.log('JSZip not available, using fallback method');
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const zipFileName = `${assetType}_${dataType}.csv`; // Mantieni come CSV se non possiamo creare ZIP
        downloadBlob(blob, zipFileName);
      }
      
    } catch (error) {
      console.error('ZIP creation error:', error);
      // Fallback: scarica come CSV normale
      const blob = new Blob([csvContent], { type: 'text/csv' });
      const csvFileName = `${assetType}_${dataType}.csv`;
      downloadBlob(blob, csvFileName);
    }
  };

  // Funzione per creare ZIP lato client da dati diretti
  const createClientSideZip = async (responseData, fileName, assetType, dataType) => {
    try {
      console.log('Creating client-side ZIP');
      
      let content = '';
      if (typeof responseData === 'string') {
        content = responseData;
      } else if (responseData.data) {
        content = typeof responseData.data === 'string' ? responseData.data : JSON.stringify(responseData.data, null, 2);
      } else {
        content = JSON.stringify(responseData, null, 2);
      }
      
      await createZipFromCSV(content, fileName, assetType, dataType);
      
    } catch (error) {
      console.error('Client-side ZIP creation failed:', error);
      // Fallback: scarica come file normale
      let fallbackContent = '';
      if (typeof responseData === 'string') {
        fallbackContent = responseData;
      } else if (responseData.data) {
        fallbackContent = typeof responseData.data === 'string' ? responseData.data : JSON.stringify(responseData.data, null, 2);
      } else {
        fallbackContent = JSON.stringify(responseData, null, 2);
      }
      
      const blob = new Blob([fallbackContent], { type: 'application/octet-stream' });
      downloadBlob(blob, fileName);
    }
  };

  // Funzione helper per scaricare un blob
  const downloadBlob = (blob, fileName) => {
    const blobUrl = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = blobUrl;
    link.download = fileName;
    link.style.display = 'none';
    
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    setTimeout(() => {
      window.URL.revokeObjectURL(blobUrl);
    }, 100);
    
    console.log('File downloaded:', fileName);
  };
{/*
  // Funzione per creare ZIP lato client da dati diretti (legacy)
  const createAndDownloadZip = async (responseData, fileName) => {
    console.log('Using legacy createAndDownloadZip function');
    return createClientSideZip(responseData, fileName, 'unknown', 'data');
  };
*/}
  // Verifica se l'utente può accedere a un tipo di dati
  const canAccessData = (assetType, dataType) => {
    if (!accessLimits) {
      console.log('No access limits loaded yet');
      return false;
    }

    const dataKey = `${assetType}_${dataType}`;
    console.log(`Checking access for: ${dataKey}`);
    console.log('User limits:', accessLimits);
    console.log('User role:', user?.role_name);

    // Se l'utente ha accesso illimitato
    if (accessLimits.unlimited_access) {
      console.log('Unlimited access granted');
      return true;
    }

    // Gli admin hanno accesso a tutto
    if (user?.role_name === 'admin') {
      console.log('Admin access granted');
      return true;
    }

    // I premium/senator hanno accesso esteso
    if (user?.role_name === 'premium' || user?.role_name === 'senator') {
      console.log('Premium/Senator access granted');
      return true; // Assumiamo che premium abbia accesso a tutto
    }

    // Gli utenti base hanno accesso limitato
    if (user?.role_name === 'base') {
      // Per ora tutti gli utenti base possono scaricare, il backend filtrerà
      console.log('Base user - limited access');
      return true; // Lascia che il backend gestisca i limiti
    }

    console.log('No access granted for unknown user type');
    return false;
  };

  // Ottieni lo stato del pulsante di download
  const getDownloadButtonState = (assetType, dataType) => {
    const downloadKey = `${assetType}_${dataType}`;
    const state = downloadStates[downloadKey];
    const hasAccess = canAccessData(assetType, dataType);

    return { state, hasAccess };
  };

  // Renderizza un pulsante di download ZIP
  const renderDownloadButton = (assetType, dataType, label) => {
    const { state, hasAccess } = getDownloadButtonState(assetType, dataType);

    if (!isAuthenticated) {
      return (
        <button 
          className="download-link login-required"
          onClick={() => navigate('/login')}
        >
          <Lock size={16} />
          Login Required - {label} ZIP ZIP
        </button>
      );
    }

    let buttonContent;
    let buttonClass = "download-link";

    switch (state) {
      case 'downloading':
        buttonContent = (
          <>
            <div className="spinner"></div>
            Creating {label} ZIP...
          </>
        );
        buttonClass += " downloading";
        break;
        
      case 'completed':
        buttonContent = (
          <>
            <CheckCircle size={16} />
            {label} ZIP Downloaded!
          </>
        );
        buttonClass += " completed";
        break;
        
      case 'error':
        buttonContent = (
          <>
            <AlertTriangle size={16} />
            Retry {label} ZIP
          </>
        );
        buttonClass += " error";
        break;
        
      default:
        buttonContent = (
          <>
            <Download size={16} />
            Download {label} ZIP
          </>
        );
    }

    return (
      <button 
        className={buttonClass}
        onClick={() => handleDownload(assetType, dataType)}
        disabled={state === 'downloading'}
        title={`Download ${label} as ZIP archive containing all CSV files`}
      >
        {buttonContent}
      </button>
    );
  };

  if (authLoading || loading) {
    return (
      <div className="free-access-page">
        <div className="free-access-container">
          <div style={{ textAlign: 'center', padding: '50px' }}>
            <div className="spinner"></div>
            <p>Loading ZIP download options...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="free-access-page">
      <div className="free-access-container">
        <h1>Direct ZIP Downloads</h1>
        
        <p className="subtitle">
          Download complete datasets as ZIP archives containing organized CSV files. 
          Each ZIP file includes all relevant data with clear file naming conventions.
          {!isAuthenticated && (
            <> Please <button onClick={() => navigate('/login')} style={{background: 'none', border: 'none', color: '#007bff', cursor: 'pointer', textDecoration: 'underline'}}>login</button> to access downloads.</>
          )}
        </p>

        {/* Mostra errori se presenti */}
        {error && (
          <div className="error-box" style={{
            background: '#ffebee',
            border: '1px solid #f44336',
            borderRadius: '8px',
            padding: '15px',
            marginBottom: '20px',
            color: '#c62828'
          }}>
            <AlertTriangle size={20} style={{ marginRight: '10px' }} />
            {error}
            <button 
              onClick={loadAccessLimits} 
              style={{
                marginLeft: '10px',
                background: '#f44336',
                color: 'white',
                border: 'none',
                padding: '5px 10px',
                borderRadius: '4px',
                cursor: 'pointer'
              }}
            >
              Retry
            </button>
          </div>
        )}

        {/* Realized Variances Section */}
        <div className="download-section">
          <h2>Realized Variances ZIP Archives</h2>
          <p className="section-description">
            Download comprehensive ZIP archives containing available realized variance measures including 
            Bipower Variation, Realized Semi-Variances, Median RV, Minimum RV, Realized Kernel, 
            and Range-based measures. Each ZIP contains multiple CSV files organized by measure type.
          </p>
          
          <div className="download-links">
            {assetTypes.map(({ key, label }) => (
              <div key={key}>
                {renderDownloadButton(key, 'variances', `${label} Variances`)}
              </div>
            ))}
          </div>
        </div>

        {/* Covariances Section */}
        <div className="download-section">
          <h2>Realized Covariances ZIP Archives</h2>
          <p className="section-description">
            Download ZIP archives containing available realized covariance matrices in long format CSV files 
            (Asset1, Asset2, Date, Covariance). Each ZIP includes all pairwise covariances 
            between assets within the same category as organized CSV files.
          </p>
          
          <div className="download-links">
            {assetTypes.map(({ key, label }) => (
              <div key={`cov-${key}`}>
                {renderDownloadButton(key, 'covariances', `${label} Covariances`)}
              </div>
            ))}
          </div>
        </div>

  {/* Documentation Link */}
          {/*
          <div className="doc-links">
            <div className="info-box">
              <h3>Archive Information</h3>
              <p>
                <strong>Archive Format:</strong> All downloads are provided as CSV files for easy handling.<br/>
                <strong>Contents:</strong> Each file contains multiple columns organized by data type.<br/>
                <strong>File Structure:</strong> Clear naming conventions and column organization within each file.<br/>
                <strong>Compression:</strong> Standard file compression for faster downloads and storage efficiency.<br/>
                <strong>Documentation:</strong> <a href="/documentation">Visit our documentation page</a> for detailed descriptions of each measure and file structure.
              </p>
            </div>
          </div>
          */}
      </div>
    </div>
  );
};

export default DownloadPage;