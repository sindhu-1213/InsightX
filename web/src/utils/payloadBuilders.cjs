/**
 * Shared payload builders for InsightX Firestore documents.
 * Ensures rules tests and web client write identical document shapes.
 */

let _serverTimestamp;
function getServerTimestamp() {
  if (_serverTimestamp) return _serverTimestamp;
  // In Node test runner (outside browser and outside production Vite bundling)
  if (typeof window === 'undefined' && process.env.NODE_ENV !== 'production') {
    try {
      const firestorePath = require.resolve('firebase/firestore', { paths: [process.cwd(), __dirname] });
      _serverTimestamp = require(firestorePath).serverTimestamp;
    } catch {
      try {
        _serverTimestamp = require('firebase/firestore').serverTimestamp;
      } catch {
        // Fallback if unavailable
      }
    }
  }
  return _serverTimestamp;
}

function buildUserProfile({ displayName, role, institution }) {
  const profile = {
    displayName: (displayName || '').trim().slice(0, 99),
    role: ['Doctor', 'Staff', 'Student'].includes(role) ? role : 'Doctor',
  };
  if (institution !== undefined && institution !== null) {
    profile.institution = String(institution).trim().slice(0, 99);
  }
  return profile;
}

function buildAnalysisRecord({ finding, label, probability, modelVersion, source, createdAt }) {
  let findingText = finding;
  if (!findingText) {
    if (label === 'Abnormal') {
      findingText = 'Possible abnormal finding';
    } else {
      findingText = 'No abnormal finding flagged';
    }
  }
  let defaultCreatedAt;
  const serverTsFn = getServerTimestamp();
  if (typeof serverTsFn === 'function') {
    defaultCreatedAt = serverTsFn();
  } else {
    defaultCreatedAt = new Date().toISOString();
  }
  return {
    finding: findingText,
    probability: Number(probability),
    modelVersion: String(modelVersion || '1.0.0').slice(0, 99),
    source: source === 'sample' ? 'sample' : 'upload',
    createdAt: createdAt !== undefined ? createdAt : defaultCreatedAt,
  };
}

module.exports = {
  buildUserProfile,
  buildAnalysisRecord,
};
