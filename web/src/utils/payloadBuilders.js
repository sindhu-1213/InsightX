import { serverTimestamp } from 'firebase/firestore';
import builders from './payloadBuilders.cjs';

// Bind web client's modular serverTimestamp to payload builders
builders.setServerTimestamp(serverTimestamp);

export const buildUserProfile = builders.buildUserProfile;
export const buildAnalysisRecord = builders.buildAnalysisRecord;
export const setServerTimestamp = builders.setServerTimestamp;

export default builders;
