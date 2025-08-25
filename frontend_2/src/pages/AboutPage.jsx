import React from "react";
import { motion } from "framer-motion";
import "../styles/about.css";

const AboutPage = () => {
  const pageVariants = {
    hidden: { opacity: 0, y: 30 },
    visible: { 
      opacity: 1, 
      y: 0, 
      transition: { duration: 0.8, staggerChildren: 0.2 } 
    },
  };

  const sectionVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0, 
      transition: { duration: 0.6 } 
    },
  };

  return (
    <div className="about-page">
      <motion.div 
        className="about-container"
        initial="hidden"
        animate="visible"
        variants={pageVariants}
      >
        <div className="about-content">
          <motion.div className="intro-section" variants={sectionVariants}>
            <h1></h1>
            <p>
              <strong>VOLARE (VOLatility Archive for Realized Estimates)</strong> was born out of a simple goal: 
              to make high-quality, research-grade realized volatility data accessible to those who need it most financial researchers, 
              quantitative analysts, and academics.
            </p>
            <p>
              VOLARE is our response to the growing demand for volatility modeling tools that are reliable and easy to use. We adopted <strong>quality, transparency, and methodological rigor</strong>.
            </p>
          </motion.div>

          <motion.section variants={sectionVariants}>
            <h2>Why We Built VOLARE</h2>
            <p>
              While working with ultra-high-frequency data, we experienced firsthand the complexity of processing, cleaning, 
              and extracting meaningful volatility estimates from raw financial feeds. We created VOLARE to streamline this 
              process — without sacrificing the methodological integrity required in academic and professional research.
            </p>
            <p>
              We aim to bridge the gap between <strong>cutting-edge research</strong> and <strong>practical usability</strong>, 
              enabling researchers to focus on insights rather than infrastructure.
            </p>
          </motion.section>

          <motion.section variants={sectionVariants}>
            <h2>Our Philosophy</h2>
            <ul>
              <li>
                <strong>Transparency</strong>: Every estimator, every dataset, every transformation is documented and reproducible.
              </li>
              <li>
                <strong>Precision</strong>: All volatility measures are based on millisecond-level data, cleaned and curated for accuracy.
              </li>
              <li>
                <strong>Accessibility</strong>: We believe powerful tools should be usable. VOLARE offers a clean interface and 
                flexible workflows for users at every level of expertise.
              </li>
            </ul>
          </motion.section>
{/*
          <motion.section variants={sectionVariants}>
            <h2>Technology Stack</h2>
            <p>VOLARE is built with a focus on performance, scalability, and clarity:</p>
            <ul>
              <li>
                <strong>Python</strong> powers the backend, including custom-built modules for high-frequency data cleaning, 
                estimator computation, and batch processing.
              </li>
              <li>
                <strong>Parquet</strong> and other efficient storage formats ensure fast I/O for large datasets.
              </li>
              <li>
                <strong>React</strong> and modern web technologies drive an intuitive, responsive frontend.
              </li>
              <li>
                <strong>Concurrency and caching</strong> mechanisms are used throughout the pipeline to optimize performance 
                without compromising data integrity.
              </li>
            </ul>
          </motion.section>
*/}
          <motion.section variants={sectionVariants}>
            <h2>A Platform for Evolving Research</h2>
            <p>
              We designed VOLARE as a work in progress. As the financial landscape evolves, the platform will expand with new volatility 
              measures, asset types, and research features guided by the needs of the community.
            </p>
          </motion.section>

          <motion.section variants={sectionVariants}>
            <h2>Get in Touch</h2>
            <p>
              If you are a researcher, developer, or institution interested in using or contributing to VOLARE, feel free to reach out. 
              We are always open to collaboration, feedback, and shared ideas.
            </p>
          </motion.section>
        </div>
      </motion.div>
    </div>
  );
};

export default AboutPage;