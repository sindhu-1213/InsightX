/**
 * Shared payload builders for InsightX Firestore documents.
 * Ensures rules tests and web client write identical document shapes.
 */

let defaultServerTimestamp;

function setServerTimestamp(fn) {
  defaultServerTimestamp = fn;
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
  let resolvedCreatedAt = createdAt;
  if (resolvedCreatedAt === undefined && typeof defaultServerTimestamp === 'function') {
    resolvedCreatedAt = defaultServerTimestamp();
  }
  return {
    finding: findingText,
    probability: Number(probability),
    modelVersion: String(modelVersion || '1.0.0').slice(0, 99),
    source: source === 'sample' ? 'sample' : 'upload',
    createdAt: resolvedCreatedAt,
  };
}

module.exports = {
  buildUserProfile,
  buildAnalysisRecord,
  setServerTimestamp,
};
