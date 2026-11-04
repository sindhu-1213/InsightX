import { serverTimestamp } from 'firebase/firestore';
import builders from './payloadBuilders.cjs';

export const buildUserProfile = builders.buildUserProfile;
export const buildAnalysisRecord = (options = {}) => {
  if (options.createdAt === undefined) {
    return builders.buildAnalysisRecord({
      ...options,
      createdAt: serverTimestamp()
    });
  }
  return builders.buildAnalysisRecord(options);
};

export default {
  buildUserProfile,
  buildAnalysisRecord
};
