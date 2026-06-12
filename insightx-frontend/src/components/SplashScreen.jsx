import React from 'react';
import './SplashScreen.css';
import logoAnimation from '../assets/logo-animation.mp4';

function SplashScreen({ onFinish }) {
  return (
    <div className="splash-container">
      <video
        className="splash-video"
        src={logoAnimation}
        autoPlay
        muted
        playsInline
        onEnded={onFinish}
        onPlay={() => console.log('video started')}
        onError={(e) => console.log('video error', e)}
      />
    </div>
  );
}

export default SplashScreen;