import React, { useRef } from 'react';
import logoAnimation from '../assets/logo-animation.mp4';

function SplashScreen({ onFinish }) {
  const videoRef = useRef(null);

  return (
    <div
      id="splash-screen-container"
      style={{
        position: 'fixed',
        inset: 0,
        width: '100vw',
        height: '100vh',
        backgroundColor: '#020b18',
        zIndex: 99999,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden'
      }}
    >
      <video
        ref={videoRef}
        id="splash-video-player"
        src={logoAnimation}
        autoPlay
        muted
        playsInline
        onEnded={onFinish}
        onError={() => onFinish?.()}
        style={{
          width: '100%',
          height: '100%',
          objectFit: 'cover',
          display: 'block'
        }}
      />

      {/* Skip Button */}
      <button
        id="skip-splash-btn"
        onClick={onFinish}
        style={{
          position: 'absolute',
          bottom: '2rem',
          right: '2rem',
          backgroundColor: 'rgba(4, 20, 40, 0.75)',
          color: '#00d4ff',
          border: '1px solid rgba(0, 212, 255, 0.4)',
          borderRadius: '9999px',
          padding: '0.5rem 1.25rem',
          fontSize: '0.85rem',
          fontFamily: 'var(--font-heading)',
          letterSpacing: '0.05em',
          cursor: 'pointer',
          backdropFilter: 'blur(8px)',
          boxShadow: '0 0 15px rgba(0, 212, 255, 0.2)',
          transition: 'all 0.2s ease',
          zIndex: 10
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.backgroundColor = 'rgba(0, 212, 255, 0.15)';
          e.currentTarget.style.borderColor = '#00d4ff';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.backgroundColor = 'rgba(4, 20, 40, 0.75)';
          e.currentTarget.style.borderColor = 'rgba(0, 212, 255, 0.4)';
        }}
      >
        Skip Intro &rarr;
      </button>
    </div>
  );
}

export default SplashScreen;
