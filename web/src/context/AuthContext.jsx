import React, { createContext, useContext, useEffect, useState } from 'react';
import {
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut,
  sendEmailVerification,
  sendPasswordResetEmail,
  updateProfile,
  onAuthStateChanged,
  getRedirectResult
} from 'firebase/auth';
import { doc, getDoc, setDoc } from 'firebase/firestore';
import {
  auth,
  db,
  googleProvider,
  isIOSOrInAppBrowser,
  signInWithPopup,
  signInWithRedirect
} from '../firebase';

const AuthContext = createContext(null);

export function getFriendlyAuthErrorMessage(error) {
  if (!error) return '';
  const code = error.code || '';
  switch (code) {
    case 'auth/user-not-found':
    case 'auth/wrong-password':
    case 'auth/invalid-credential':
      return 'Invalid email or password. Please verify your credentials.';
    case 'auth/email-already-in-use':
      return 'An account with this email address already exists. Please sign in instead.';
    case 'auth/weak-password':
      return 'Password is too weak. Please choose at least 6 characters.';
    case 'auth/invalid-email':
      return 'Please enter a valid email address.';
    case 'auth/user-disabled':
      return 'This practitioner account has been deactivated. Please contact support.';
    case 'auth/too-many-requests':
      return 'Access temporarily blocked due to repeated failed attempts. Please try again shortly.';
    case 'auth/popup-closed-by-user':
      return 'Google sign-in was cancelled before completion.';
    case 'auth/popup-blocked':
      return 'The sign-in popup was blocked by your browser. Please allow popups or try redirect.';
    case 'auth/network-request-failed':
      return 'Network connection error. Please verify your connectivity.';
    default:
      return error.message || 'An unexpected authentication error occurred.';
  }
}

export function AuthProvider({ children }) {
  const [currentUser, setCurrentUser] = useState(null);
  const [userProfile, setUserProfile] = useState(null);
  const [loading, setLoading] = useState(true);

  // Sync profile from Firestore or write pending profile if email just got verified
  const fetchOrSyncProfile = async (user) => {
    if (!user) {
      setUserProfile(null);
      return null;
    }

    const userDocRef = doc(db, 'users', user.uid);
    try {
      const snap = await getDoc(userDocRef);
      if (snap.exists()) {
        const data = snap.data();
        setUserProfile(data);
        return data;
      }

      // Check if there is a pending profile saved locally (e.g. from initial signup)
      const pendingKey = `insightx_profile_${user.uid}`;
      const cached = localStorage.getItem(pendingKey);
      if (cached) {
        const parsed = JSON.parse(cached);
        await setDoc(userDocRef, parsed);
        localStorage.removeItem(pendingKey);
        setUserProfile(parsed);
        return parsed;
      }
    } catch (err) {
      console.warn('Firestore profile sync note:', err.message);
    }
    return null;
  };

  useEffect(() => {
    // Handle redirect outcome for iOS/in-app browsers
    getRedirectResult(auth)
      .then(async (result) => {
        if (result && result.user) {
          const u = result.user;
          const userDocRef = doc(db, 'users', u.uid);
          const snap = await getDoc(userDocRef).catch(() => null);
          if (!snap || !snap.exists()) {
            const initialProfile = {
              displayName: (u.displayName || 'Practitioner').slice(0, 99),
              role: 'Doctor',
              institution: 'Independent Practitioner'
            };
            await setDoc(userDocRef, initialProfile).catch(() => {});
            setUserProfile(initialProfile);
          }
        }
      })
      .catch((err) => {
        console.warn('Google redirect result error:', err);
      });

    const unsubscribe = onAuthStateChanged(auth, async (user) => {
      setCurrentUser(user);
      if (user) {
        await fetchOrSyncProfile(user);
      } else {
        setUserProfile(null);
      }
      setLoading(false);
    });

    return unsubscribe;
  }, []);

  const signup = async (email, password, { displayName, role, institution }) => {
    const res = await createUserWithEmailAndPassword(auth, email, password);
    const user = res.user;

    const trimmedName = (displayName || 'Practitioner').trim().slice(0, 99);
    const validRole = ['Doctor', 'Staff', 'Student'].includes(role) ? role : 'Doctor';
    const trimmedInst = (institution || 'Independent Practitioner').trim().slice(0, 99);

    const profileData = {
      displayName: trimmedName,
      role: validRole,
      institution: trimmedInst
    };

    // Update Firebase Auth profile
    try {
      await updateProfile(user, { displayName: trimmedName });
    } catch (e) {
      console.warn('Could not update profile display name:', e);
    }

    // Send verification email
    try {
      await sendEmailVerification(user);
    } catch (e) {
      console.warn('Could not send email verification:', e);
    }

    // Write users/{uid} to Firestore or cache for verification
    try {
      await setDoc(doc(db, 'users', user.uid), profileData);
      setUserProfile(profileData);
    } catch (_e) {
      // If firestore rules require verified email before writing, persist to localStorage
      localStorage.setItem(`insightx_profile_${user.uid}`, JSON.stringify(profileData));
    }

    setCurrentUser({ ...user, displayName: trimmedName });
    return user;
  };

  const login = async (email, password) => {
    const res = await signInWithEmailAndPassword(auth, email, password);
    await fetchOrSyncProfile(res.user);
    return res.user;
  };

  const signInWithGoogle = async (role = 'Doctor', institution = 'Independent Practitioner') => {
    if (isIOSOrInAppBrowser()) {
      // Use redirect on iOS Safari / in-app browsers
      return await signInWithRedirect(auth, googleProvider);
    }

    // Use popup on standard desktop and Android Chrome
    const res = await signInWithPopup(auth, googleProvider);
    const u = res.user;

    // Check or create Firestore profile
    const userDocRef = doc(db, 'users', u.uid);
    try {
      const snap = await getDoc(userDocRef);
      if (!snap.exists()) {
        const profileData = {
          displayName: (u.displayName || 'Practitioner').slice(0, 99),
          role: ['Doctor', 'Staff', 'Student'].includes(role) ? role : 'Doctor',
          institution: (institution || 'Independent Practitioner').slice(0, 99)
        };
        await setDoc(userDocRef, profileData);
        setUserProfile(profileData);
      } else {
        setUserProfile(snap.data());
      }
    } catch (e) {
      console.warn('Could not save Google user profile:', e);
    }

    return u;
  };

  const logout = async () => {
    await signOut(auth);
    setUserProfile(null);
  };

  const resetPassword = async (email) => {
    return await sendPasswordResetEmail(auth, email);
  };

  const sendVerification = async () => {
    if (auth.currentUser) {
      return await sendEmailVerification(auth.currentUser);
    }
  };

  const reloadUser = async () => {
    if (auth.currentUser) {
      await auth.currentUser.reload();
      const updated = auth.currentUser;
      setCurrentUser({ ...updated });
      await fetchOrSyncProfile(updated);
      return updated;
    }
    return null;
  };

  const getIdToken = async (forceRefresh = false) => {
    if (auth.currentUser) {
      return await auth.currentUser.getIdToken(forceRefresh);
    }
    return null;
  };

  const value = {
    currentUser,
    userProfile,
    loading,
    signup,
    login,
    logout,
    resetPassword,
    sendVerification,
    reloadUser,
    getIdToken,
    signInWithGoogle
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
