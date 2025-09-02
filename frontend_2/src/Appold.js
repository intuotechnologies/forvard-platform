import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import HomePage from "./pages/HomePage";
import LoginPage from "./pages/LoginPage";
import SignUpPage from "./pages/SignUpPage";
import Dashboard from "./pages/Dashboard";
import GraphPage from "./pages/GraphPage";
import Footer from "./components/Footer";
import Navbar from "./components/Navbar";
import AccessPage from "./pages/AccessPage"; 
import DownloadPage from "./pages/DownloadPage";
import DPage from "./pages/DPage";
import UnifiedPage from "./pages/UnifiedPage"; 

import "./styles/global.css";

function App() {
  return (
    <>
      <Router>
        <Navbar />
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/graph" element={<GraphPage />} /> 
          <Route path="/login" element={<LoginPage />} />
          <Route path="/signup" element={<SignUpPage />} />
          <Route path="/access" element={<AccessPage />} />
          <Route path="/download" element={<DownloadPage />} />
          <Route path="/uni" element={<UnifiedPage />} />
          <Route path="/d" element={<DPage />} />
        </Routes>
      </Router>
      <Footer />
    </>
  );
}

export default App;


