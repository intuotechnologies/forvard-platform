import React from "react";
import "../styles/footer.css";

const Footer = () => {
  return (
    <footer className="footer">
      <div className="footer-section footer-left">
        <img src="/logo_footer.png" alt="Logo" className="footer-logo" />
      </div>
      <div className="footer-section footer-center">
        <a href="/privacy">Privacy Policy</a> | <a href="/terms">Terms & Conditions</a>
      </div>
      <div className="footer-section footer-right">
        <a href="mailto:info@email.com">info@email.com</a>
      </div>
    </footer>
  );
};

export default Footer;
