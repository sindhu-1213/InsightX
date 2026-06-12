import React from 'react';

function HeroSection() {
  return (
    <section
      style={{
        minHeight: '80vh',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        textAlign: 'center',
        padding: '0 20px'
      }}
    >
      <h1
        style={{
          fontSize: '4rem',
          color: '#ffffff',
          marginBottom: '20px',
          fontFamily: 'Orbitron, sans-serif'
        }}
      >
        AI-Powered Chest X-Ray Analysis
      </h1>

      <p
        style={{
          maxWidth: '800px',
          color: '#8ab4c9',
          fontSize: '1.2rem',
          lineHeight: '1.8'
        }}
      >
        InsightX combines deep learning, medical text analysis,
        and explainable AI to assist healthcare professionals
        in detecting abnormalities from chest radiographs.
      </p>
      <button
  style={{
    marginTop: '30px',
    padding: '14px 32px',
    backgroundColor: '#00d4ff',
    color: '#020b18',
    border: 'none',
    borderRadius: '8px',
    fontSize: '1rem',
    fontWeight: 'bold',
    cursor: 'pointer'
  }}
>
  Get Started
</button>
    </section>
  );
}

export default HeroSection;