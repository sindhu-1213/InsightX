import { initializeApp, getApps, getApp } from 'firebase/app';
import {
  getAuth,
  GoogleAuthProvider,
  signInWithPopup,
  signInWithRedirect,
  getRedirectResult
} from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';

const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY || 'AIzaSyFakeKeyExampleForDev12345',
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN || 'insightx-demo.firebaseapp.com',
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID || 'insightx-demo',
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET || 'insightx-demo.firebasestorage.app',
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID || '123456789012',
  appId: import.meta.env.VITE_FIREBASE_APP_ID || '1:123456789012:web:abcdef1234567890'
};

const app = !getApps().length ? initializeApp(firebaseConfig) : getApp();
export const auth = getAuth(app);
export const db = getFirestore(app);
export const googleProvider = new GoogleAuthProvider();
googleProvider.setCustomParameters({ prompt: 'select_account' });

/**
 * Detect iOS Safari or in-app webviews where popup windows frequently fail or get blocked.
 */
export function isIOSOrInAppBrowser() {
  if (typeof window === 'undefined' || !window.navigator) return false;
  const ua = window.navigator.userAgent || navigator.vendor || window.opera || '';
  const isIOS =
    /iPad|iPhone|iPod/.test(ua) ||
    (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
  const isInApp =
    /FBAN|FBAV|Instagram|Line|Twitter|Snapchat|MicroMessenger|WebView|wv/i.test(ua);
  const isIOSSafari = isIOS && /Safari/i.test(ua) && !/CriOS|FxiOS|EdgiOS/i.test(ua);
  return isIOS || isIOSSafari || isInApp;
}

export { signInWithPopup, signInWithRedirect, getRedirectResult };
export default app;
