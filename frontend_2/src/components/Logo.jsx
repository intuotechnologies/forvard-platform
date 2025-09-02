import React from 'react';
import { Link } from 'react-router-dom';
import '../styles/logo.css';
import logoImage from '/public/logo.png';

const Logo = () => {
  return (
    <Link to="/" className="logo">
      <img src={logoImage} alt="VOLARE Logo" className="logo-image" />
    </Link>
  );
};

export default Logo;
