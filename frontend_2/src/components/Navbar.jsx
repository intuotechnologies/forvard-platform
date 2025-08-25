// src/components/Navbar.jsx

import React, { useState, useRef, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { FaUserCircle } from "react-icons/fa";
import { User, LogOut } from "lucide-react";
import { useAuth } from "../contexts/AuthContext";
import "../styles/navbar.css";

const Navbar = () => {
  const [showUserMenu, setShowUserMenu] = useState(false);
  const menuRef = useRef(null);
  const navigate = useNavigate();
  
  const { isAuthenticated, user, logout } = useAuth();

  // Chiudi il menu se clicchi fuori
  useEffect(() => {
    function handleClickOutside(event) {
      if (menuRef.current && !menuRef.current.contains(event.target)) {
        setShowUserMenu(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleUserIconClick = () => {
    if (isAuthenticated) {
      setShowUserMenu(!showUserMenu);
    } else {
      navigate('/login');
    }
  };

  const handleLogout = () => {
    logout();
    setShowUserMenu(false);
    navigate('/');
  };

  return (
    <nav className="navbar">
      {/* Left: Logo */}
      <div className="navbar-left">
        <Link to="/" className="logo">
          <img src="/logo.png" alt="VOLARE Logo" className="logo-image" />
        </Link>
      </div>

      {/* Center: Navigation Links */}
      <div className="navbar-center">
        <Link to="/documentation">Documentation</Link>
        <Link to="/about">About</Link>
        <Link to="/contact">Contact</Link>
      </div>

      {/* Right: User Icon */}
      <div className="navbar-right" ref={menuRef}>
        <div className="user-container">
          <FaUserCircle 
            className="user-icon" 
            onClick={handleUserIconClick}
          />
          
          {/* Dropdown menu - appare solo se autenticato */}
          {isAuthenticated && showUserMenu && (
            <div className="user-dropdown">
              <div className="dropdown-item user-info">
                <User size={16} />
                {user?.email}
              </div>
              <div className="dropdown-item logout" onClick={handleLogout}>
                <LogOut size={16} />
                Logout
              </div>
            </div>
          )}
        </div>
      </div>
    </nav>
  );
};

export default Navbar;