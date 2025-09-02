import React from "react";
import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import "../styles/home.css";

const sectionVariants = {
  hidden: { opacity: 0, y: 40 },
  visible: { 
    opacity: 1, 
    y: 0, 
    transition: { 
      duration: 0.8,
      staggerChildren: 0.2 // This will stagger child animations
    } 
  },
};

// Create variants for children elements
const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: { 
    opacity: 1, 
    y: 0,
    transition: { duration: 0.5 }
  }
};

const HomePage = () => {
  return (
    <div className="homepage">
      {/* Section 1 - Hero */}
      <section className="section hero full-screen">
        <motion.div
          className="container hero-grid"
          initial="hidden"
          animate="visible"
          viewport={{ once: false, amount: 0.3 }}
          variants={sectionVariants}
        >
          <motion.div className="hero-content" variants={itemVariants}>
            <motion.h1 className="hero-title" variants={itemVariants}>
              Unlock the Power of Data
            </motion.h1>
            <motion.p className="hero-subtitle" variants={itemVariants}>
              your go-to portal for realized volatility and realized covariance datasets across a large number of assets (stocks, exchange rates and futures).
            </motion.p>
            <motion.div 
              className="hero-buttons"
              variants={itemVariants}
              style={{ 
                display: 'flex', 
                gap: '20px', 
                justifyContent: 'center',
                flexWrap: 'wrap'
              }}
            >
              <Link to="/dashboard" className="cta-button">
              Visualize Data
            </Link>
              <Link to="/download" className="cta-button">
                Download Data
              </Link>
            </motion.div>
          </motion.div>
          <motion.div className="hero-image" variants={itemVariants}>
            <img
              src="/hero-graphic.png"
              alt="Data visualization illustration"
              className="illustration"
            />
          </motion.div>
        </motion.div>
      </section>

      {/* Section 2 - Mission */}
      <section className="section mission full-screen">
        <motion.div
          className="container narrow"
          initial="hidden"
          whileInView="visible"
          viewport={{ once: false, amount: 0.3 }}
          variants={sectionVariants}
        >
          <motion.p 
            className="mission-text"
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            transition={{ 
              staggerChildren: 0.05,
              duration: 1.5
            }}
          >
            {/* Split text into words for animation */}
            {"We simplify the use of realized measures for researchers and academics. Get instant free access to clean datasets and visualization tools."
              .split(" ")
              .map((word, i) => (
                <motion.span
                  key={i}
                  style={{ display: "inline-block", marginRight: "0.4em" }}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: i * 0.05 }}
                >
                  {word}
                </motion.span>
              ))}
          </motion.p>
        </motion.div>
      </section>

      {/* Section 3 - Features */}
      <section className="section features full-screen">
        <motion.div
          className="container features-list"
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, amount: 0.5 }}
          variants={sectionVariants}
        >
          <div className="feature-card">
            <h3>High-frequency Data</h3>
            <p>We have cleaned millisecond-level financial data to deliver maximum accuracy and variety.</p>
          </div>
          <div className="feature-card">
            <h3>Ready-to-use Datasets</h3>
            <p>Choose and download the format that best fits your workflow.</p>
          </div>
          <div className="feature-card">
            <h3>Measures Comparison</h3>
            <p>Visualize and compare multiple realized volatility measures.</p>
          </div>
        </motion.div>
      </section>

      {/* Section 4 - Final CTA */}
      <section className="section final-cta not-full">
        <motion.div
          className="container narrow"
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, amount: 0.5 }}
          variants={sectionVariants}
        >
          <h2 className="cta-heading">Ready to explore?</h2>
          <div 
            style={{ 
              display: 'flex', 
              gap: '20px', 
              justifyContent: 'center',
              flexWrap: 'wrap'
            }}
          >
            <Link to="/dashboard" className="cta-button">
              Visualize Data
            </Link>
            <Link to="/download" className="cta-button">
              Download Data
            </Link>
            
          </div>
        </motion.div>
      </section>
    </div>
  );
};

export default HomePage;