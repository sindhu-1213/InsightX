/**
 * Firestore Security Rules Unit Tests using @firebase/rules-unit-testing.
 *
 * Verifies:
 * - Only signed-in users with a verified email can read/write their own data.
 * - Complete denial of access to other users' data.
 * - Strict schema validation on users/{uid} (displayName required < 100 chars, role in [Doctor, Staff, Student], institution optional < 100 chars).
 * - Strict schema validation on analyses/{id} (finding in ['Possible abnormal finding', 'No abnormal finding flagged'], probability in [0, 1], modelVersion, source in ['upload', 'sample'], createdAt).
 * - Immutability: analyses documents cannot be updated after creation.
 */

const {
  initializeTestEnvironment,
  assertFails,
  assertSucceeds,
} = require("@firebase/rules-unit-testing");
const assert = require("assert");
const {
  doc,
  getDoc,
  setDoc,
  updateDoc,
  deleteDoc,
  Timestamp,
  serverTimestamp,
} = require("firebase/firestore");
const fs = require("fs");
const path = require("path");
const {
  buildUserProfile,
  buildAnalysisRecord,
  setServerTimestamp,
} = require("../../web/src/utils/payloadBuilders.cjs");

// Provide test runner's Firestore serverTimestamp to payload builders
setServerTimestamp(serverTimestamp);

const PROJECT_ID = "insightx-test";
const RULES_PATH = path.resolve(__dirname, "../../firestore.rules");

describe("InsightX Firestore Security Rules", function () {
  this.timeout(15000);

  let testEnv;

  before(async () => {
    const rules = fs.readFileSync(RULES_PATH, "utf8");
    testEnv = await initializeTestEnvironment({
      projectId: PROJECT_ID,
      firestore: {
        rules: rules,
        host: process.env.FIRESTORE_EMULATOR_HOST ? process.env.FIRESTORE_EMULATOR_HOST.split(":")[0] : "127.0.0.1",
        port: process.env.FIRESTORE_EMULATOR_HOST ? parseInt(process.env.FIRESTORE_EMULATOR_HOST.split(":")[1]) : 8080,
      },
    });
  });

  after(async () => {
    if (testEnv) {
      await testEnv.cleanup();
    }
  });

  beforeEach(async () => {
    if (testEnv) {
      await testEnv.clearFirestore();
    }
  });

  // Helper to obtain user firestore contexts
  function getAuthedDb(uid, emailVerified = true) {
    return testEnv.authenticatedContext(uid, {
      email_verified: emailVerified,
      email: `${uid}@example.com`,
    }).firestore();
  }

  function getUnauthedDb() {
    return testEnv.unauthenticatedContext().firestore();
  }

  // -------------------------------------------------------------------------
  // 1. Authentication & Email Verification Tests
  // -------------------------------------------------------------------------

  describe("Authentication & Email Verification", () => {
    it("denies unauthenticated read and write to users/{uid}", async () => {
      const db = getUnauthedDb();
      const userRef = doc(db, "users/user_123");

      await assertFails(getDoc(userRef));
      await assertFails(
        setDoc(userRef, {
          displayName: "Dr. Alice",
          role: "Doctor",
          institution: "City Hospital",
        })
      );
    });

    it("denies authenticated users with unverified email", async () => {
      const db = getAuthedDb("user_unverified", false);
      const userRef = doc(db, "users/user_unverified");

      await assertFails(getDoc(userRef));
      await assertFails(
        setDoc(userRef, {
          displayName: "Unverified User",
          role: "Student",
          institution: "University",
        })
      );
    });

    it("allows verified user to read and write their own profile", async () => {
      const db = getAuthedDb("user_verified", true);
      const userRef = doc(db, "users/user_verified");

      await assertSucceeds(
        setDoc(userRef, {
          displayName: "Dr. Jane Smith",
          role: "Doctor",
          institution: "IU Health University Hospital",
        })
      );
      await assertSucceeds(getDoc(userRef));
    });

    it("denies access to other users' documents (cross-user isolation)", async () => {
      const aliceDb = getAuthedDb("alice", true);
      const bobDb = getAuthedDb("bob", true);

      // Alice sets up her own profile
      await assertSucceeds(
        setDoc(doc(aliceDb, "users/alice"), {
          displayName: "Alice",
          role: "Doctor",
          institution: "Hospital A",
        })
      );

      // Bob cannot read or overwrite Alice's profile
      await assertFails(getDoc(doc(bobDb, "users/alice")));
      await assertFails(
        setDoc(doc(bobDb, "users/alice"), {
          displayName: "Hacked",
          role: "Staff",
          institution: "Hospital B",
        })
      );
    });
  });

  // -------------------------------------------------------------------------
  // 2. User Document Schema Validation
  // -------------------------------------------------------------------------

  describe("users/{uid} Schema Validation", () => {
    it("allows valid roles: Doctor, Staff, Student", async () => {
      for (const role of ["Doctor", "Staff", "Student"]) {
        const uid = `user_${role.toLowerCase()}`;
        const db = getAuthedDb(uid, true);
        await assertSucceeds(
          setDoc(doc(db, `users/${uid}`), {
            displayName: `Valid ${role}`,
            role: role,
            institution: "Medical Center",
          })
        );
      }
    });

    it("rejects unauthorized role values (e.g. Admin, Radiologist)", async () => {
      const db = getAuthedDb("user_bad_role", true);
      await assertFails(
        setDoc(doc(db, "users/user_bad_role"), {
          displayName: "Super Admin",
          role: "Admin",
          institution: "Hospital",
        })
      );
    });

    it("rejects unexpected fields (e.g. isAdmin, permissions)", async () => {
      const db = getAuthedDb("user_extra_fields", true);
      await assertFails(
        setDoc(doc(db, "users/user_extra_fields"), {
          displayName: "Dr. Dave",
          role: "Doctor",
          institution: "Clinic",
          isAdmin: true,
        })
      );
    });

    it("rejects displayName or institution >= 100 characters", async () => {
      const db = getAuthedDb("user_long_string", true);
      const longName = "A".repeat(101);

      await assertFails(
        setDoc(doc(db, "users/user_long_string"), {
          displayName: longName,
          role: "Doctor",
          institution: "Clinic",
        })
      );

      const longInst = "B".repeat(101);
      await assertFails(
        setDoc(doc(db, "users/user_long_string"), {
          displayName: "Dr. Dave",
          role: "Doctor",
          institution: longInst,
        })
      );
    });
  });

  // -------------------------------------------------------------------------
  // 3. Analyses Collection Schema Validation & Immutability
  // -------------------------------------------------------------------------

  describe("users/{uid}/analyses/{id} Schema & Immutability", () => {
    it("allows creating valid analysis document", async () => {
      const db = getAuthedDb("doctor_uid", true);
      const analysisRef = doc(db, "users/doctor_uid/analyses/analysis_001");

      await assertSucceeds(
        setDoc(analysisRef, {
          finding: "No abnormal finding flagged",
          probability: 0.0821,
          modelVersion: "1.0.0",
          source: "upload",
          createdAt: serverTimestamp(),
        })
      );
      await assertSucceeds(getDoc(analysisRef));
    });

    it("prevents updating analyses documents after creation (immutable)", async () => {
      const db = getAuthedDb("doctor_uid", true);
      const analysisRef = doc(db, "users/doctor_uid/analyses/analysis_002");

      // 1. Create is allowed
      await assertSucceeds(
        setDoc(analysisRef, {
          finding: "Possible abnormal finding",
          probability: 0.9412,
          modelVersion: "1.0.0",
          source: "upload",
          createdAt: serverTimestamp(),
        })
      );

      // 2. Update is strictly denied
      await assertFails(
        updateDoc(analysisRef, {
          finding: "No abnormal finding flagged",
          probability: 0.12,
        })
      );
    });

    it("denies probability out of [0, 1] range", async () => {
      const db = getAuthedDb("doctor_uid", true);
      const refHigh = doc(db, "users/doctor_uid/analyses/analysis_high");
      const refLow = doc(db, "users/doctor_uid/analyses/analysis_low");

      await assertFails(
        setDoc(refHigh, {
          finding: "Possible abnormal finding",
          probability: 1.5,
          modelVersion: "1.0.0",
          source: "upload",
          createdAt: serverTimestamp(),
        })
      );

      await assertFails(
        setDoc(refLow, {
          finding: "No abnormal finding flagged",
          probability: -0.1,
          modelVersion: "1.0.0",
          source: "upload",
          createdAt: serverTimestamp(),
        })
      );
    });

    it("denies invalid finding values (must be 'Possible abnormal finding' or 'No abnormal finding flagged')", async () => {
      const db = getAuthedDb("doctor_uid", true);
      const analysisRef = doc(db, "users/doctor_uid/analyses/analysis_bad_finding");

      await assertFails(
        setDoc(analysisRef, {
          finding: "Pneumonia",
          probability: 0.85,
          modelVersion: "1.0.0",
          source: "upload",
          createdAt: serverTimestamp(),
        })
      );
    });

    it("denies extra fields in analyses documents", async () => {
      const db = getAuthedDb("doctor_uid", true);
      const analysisRef = doc(db, "users/doctor_uid/analyses/analysis_extra");

      await assertFails(
        setDoc(analysisRef, {
          finding: "No abnormal finding flagged",
          probability: 0.05,
          modelVersion: "1.0.0",
          source: "upload",
          createdAt: serverTimestamp(),
          diagnosisNotes: "Patient has cough",
        })
      );
    });

    it("denies cross-user access to analyses subcollection", async () => {
      const aliceDb = getAuthedDb("alice", true);
      const bobDb = getAuthedDb("bob", true);

      await assertSucceeds(
        setDoc(doc(aliceDb, "users/alice/analyses/ana_1"), {
          finding: "No abnormal finding flagged",
          probability: 0.05,
          modelVersion: "1.0.0",
          source: "upload",
          createdAt: serverTimestamp(),
        })
      );

      await assertFails(getDoc(doc(bobDb, "users/alice/analyses/ana_1")));
      await assertFails(
        setDoc(doc(bobDb, "users/alice/analyses/ana_1"), {
          finding: "Possible abnormal finding",
          probability: 0.99,
          modelVersion: "1.0.0",
          source: "upload",
          createdAt: serverTimestamp(),
        })
      );
    });

    it("denies signed-out user on analyses subcollection and unlisted paths", async () => {
      const db = getUnauthedDb();
      // Analyses subcollection
      await assertFails(getDoc(doc(db, "users/user_123/analyses/ana_1")));
      await assertFails(
        setDoc(doc(db, "users/user_123/analyses/ana_1"), {
          finding: "No abnormal finding flagged",
          probability: 0.1,
          modelVersion: "1.0.0",
          source: "upload",
          createdAt: serverTimestamp(),
        })
      );
      await assertFails(deleteDoc(doc(db, "users/user_123/analyses/ana_1")));

      // Unlisted arbitrary path
      await assertFails(getDoc(doc(db, "anything/else")));
      await assertFails(setDoc(doc(db, "anything/else"), { data: "test" }));
    });

    it("denies signed-in unverified user on analyses and unlisted paths", async () => {
      const db = getAuthedDb("user_unverified", false);
      // Analyses subcollection
      await assertFails(getDoc(doc(db, "users/user_unverified/analyses/ana_1")));
      await assertFails(
        setDoc(doc(db, "users/user_unverified/analyses/ana_1"), {
          finding: "No abnormal finding flagged",
          probability: 0.1,
          modelVersion: "1.0.0",
          source: "upload",
          createdAt: serverTimestamp(),
        })
      );
      await assertFails(deleteDoc(doc(db, "users/user_unverified/analyses/ana_1")));

      // Unlisted arbitrary path
      await assertFails(getDoc(doc(db, "anything/else")));
      await assertFails(setDoc(doc(db, "anything/else"), { data: "test" }));
    });

    it("denies user A deleting user B's profile or analyses", async () => {
      const aliceDb = getAuthedDb("alice", true);
      const bobDb = getAuthedDb("bob", true);

      // Alice creates profile and analysis
      await assertSucceeds(
        setDoc(doc(aliceDb, "users/alice"), {
          displayName: "Dr. Alice",
          role: "Doctor",
          institution: "Hospital A",
        })
      );
      await assertSucceeds(
        setDoc(doc(aliceDb, "users/alice/analyses/ana_1"), {
          finding: "No abnormal finding flagged",
          probability: 0.05,
          modelVersion: "1.0.0",
          source: "upload",
          createdAt: serverTimestamp(),
        })
      );

      // Bob cannot delete Alice's profile or analysis
      await assertFails(deleteDoc(doc(bobDb, "users/alice")));
      await assertFails(deleteDoc(doc(bobDb, "users/alice/analyses/ana_1")));
    });

    it("denies verified users accessing any unlisted path (default deny)", async () => {
      const db = getAuthedDb("doctor_user", true);

      await assertFails(getDoc(doc(db, "anything/else")));
      await assertFails(setDoc(doc(db, "anything/else"), { key: "value" }));
      await assertFails(getDoc(doc(db, "audit_logs/log_001")));
      await assertFails(setDoc(doc(db, "audit_logs/log_001"), { action: "login" }));
    });

    it("rejects wrong types and empty displayName for users/{uid} fields", async () => {
      const db = getAuthedDb("user_type_checks", true);
      const userRef = doc(db, "users/user_type_checks");

      // Wrong type for displayName (number)
      await assertFails(
        setDoc(userRef, {
          displayName: 12345,
          role: "Doctor",
          institution: "Hospital",
        })
      );

      // Wrong type for role (boolean or number)
      await assertFails(
        setDoc(userRef, {
          displayName: "Dr. Test",
          role: 99,
          institution: "Hospital",
        })
      );

      // Wrong type for institution (array)
      await assertFails(
        setDoc(userRef, {
          displayName: "Dr. Test",
          role: "Doctor",
          institution: ["Hospital"],
        })
      );

      // Empty string for displayName
      await assertFails(
        setDoc(userRef, {
          displayName: "",
          role: "Doctor",
          institution: "Hospital",
        })
      );

      // Missing required field (e.g. missing role)
      await assertFails(
        setDoc(userRef, {
          displayName: "Dr. Test",
          institution: "Hospital",
        })
      );
    });

    it("rejects extra image data fields in analyses documents", async () => {
      const db = getAuthedDb("doctor_uid", true);

      // Rejects imageData field
      await assertFails(
        setDoc(doc(db, "users/doctor_uid/analyses/ana_img1"), {
          finding: "No abnormal finding flagged",
          probability: 0.1,
          modelVersion: "1.0.0",
          source: "upload",
          createdAt: serverTimestamp(),
          imageData: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
        })
      );

      // Rejects rawImage field
      await assertFails(
        setDoc(doc(db, "users/doctor_uid/analyses/ana_img2"), {
          finding: "Possible abnormal finding",
          probability: 0.9,
          modelVersion: "1.0.0",
          source: "upload",
          createdAt: serverTimestamp(),
          rawImage: "base64encodedbytes",
        })
      );
    });

    it("rejects wrong types and missing fields in analyses documents", async () => {
      const db = getAuthedDb("doctor_uid", true);

      // probability as string instead of number
      await assertFails(
        setDoc(doc(db, "users/doctor_uid/analyses/ana_bad_prob_type"), {
          finding: "No abnormal finding flagged",
          probability: "0.5",
          modelVersion: "1.0.0",
          source: "upload",
          createdAt: serverTimestamp(),
        })
      );

      // finding as number instead of string
      await assertFails(
        setDoc(doc(db, "users/doctor_uid/analyses/ana_bad_finding_type"), {
          finding: 1,
          probability: 0.5,
          modelVersion: "1.0.0",
          source: "upload",
          createdAt: serverTimestamp(),
        })
      );

      // source as invalid value
      await assertFails(
        setDoc(doc(db, "users/doctor_uid/analyses/ana_bad_source_type"), {
          finding: "No abnormal finding flagged",
          probability: 0.5,
          modelVersion: "1.0.0",
          source: "invalid_source",
          createdAt: serverTimestamp(),
        })
      );

      // modelVersion >= 100 characters
      await assertFails(
        setDoc(doc(db, "users/doctor_uid/analyses/ana_long_model_version"), {
          finding: "No abnormal finding flagged",
          probability: 0.5,
          modelVersion: "V".repeat(101),
          source: "upload",
          createdAt: serverTimestamp(),
        })
      );

      // Missing modelVersion
      await assertFails(
        setDoc(doc(db, "users/doctor_uid/analyses/ana_missing_field"), {
          finding: "No abnormal finding flagged",
          probability: 0.5,
          source: "upload",
          createdAt: serverTimestamp(),
        })
      );
    });

    it("allows a verified user to delete their own analyses", async () => {
      const db = getAuthedDb("doctor_uid", true);
      const analysisRef = doc(db, "users/doctor_uid/analyses/analysis_to_delete");

      // 1. Create analysis
      await assertSucceeds(
        setDoc(analysisRef, {
          finding: "No abnormal finding flagged",
          probability: 0.15,
          modelVersion: "1.0.0",
          source: "upload",
          createdAt: serverTimestamp(),
        })
      );

      // 2. Delete analysis
      await assertSucceeds(deleteDoc(analysisRef));

      // 3. Confirm deletion
      const snap = await getDoc(analysisRef);
      if (snap.exists()) {
        throw new Error("Expected document to be deleted");
      }
    });

    it("allows createdAt with serverTimestamp() and current Timestamp", async () => {
      const db = getAuthedDb("doctor_uid", true);

      // serverTimestamp()
      await assertSucceeds(
        setDoc(doc(db, "users/doctor_uid/analyses/ana_server_ts"), {
          finding: "No abnormal finding flagged",
          probability: 0.2,
          modelVersion: "1.0.0",
          source: "upload",
          createdAt: serverTimestamp(),
        })
      );

      // Timestamp.now()
      await assertSucceeds(
        setDoc(doc(db, "users/doctor_uid/analyses/ana_now_ts"), {
          finding: "Possible abnormal finding",
          probability: 0.8,
          modelVersion: "1.0.0",
          source: "sample",
          createdAt: Timestamp.now(),
        })
      );
    });

    it("rejects createdAt timestamps far in the past or far in the future", async () => {
      const db = getAuthedDb("doctor_uid", true);

      // 1 day in the past (Timestamp)
      const pastTs = Timestamp.fromDate(new Date(Date.now() - 24 * 60 * 60 * 1000));
      await assertFails(
        setDoc(doc(db, "users/doctor_uid/analyses/ana_past_ts"), {
          finding: "No abnormal finding flagged",
          probability: 0.2,
          modelVersion: "1.0.0",
          source: "upload",
          createdAt: pastTs,
        })
      );

      // 1 day in the future (Timestamp)
      const futureTs = Timestamp.fromDate(new Date(Date.now() + 24 * 60 * 60 * 1000));
      await assertFails(
        setDoc(doc(db, "users/doctor_uid/analyses/ana_future_ts"), {
          finding: "No abnormal finding flagged",
          probability: 0.2,
          modelVersion: "1.0.0",
          source: "upload",
          createdAt: futureTs,
        })
      );

      // Current ISO string (strings are now strictly rejected; only Firestore timestamps are permitted)
      await assertFails(
        setDoc(doc(db, "users/doctor_uid/analyses/ana_current_iso_str"), {
          finding: "No abnormal finding flagged",
          probability: 0.2,
          modelVersion: "1.0.0",
          source: "upload",
          createdAt: new Date().toISOString(),
        })
      );

      // ISO string far in the past (year 2020)
      await assertFails(
        setDoc(doc(db, "users/doctor_uid/analyses/ana_past_str"), {
          finding: "No abnormal finding flagged",
          probability: 0.2,
          modelVersion: "1.0.0",
          source: "upload",
          createdAt: "2020-01-01T00:00:00.000Z",
        })
      );

      // ISO string far in the future (year 2035)
      await assertFails(
        setDoc(doc(db, "users/doctor_uid/analyses/ana_future_str"), {
          finding: "No abnormal finding flagged",
          probability: 0.2,
          modelVersion: "1.0.0",
          source: "upload",
          createdAt: "2035-01-01T00:00:00.000Z",
        })
      );

      // Invalid non-date string
      await assertFails(
        setDoc(doc(db, "users/doctor_uid/analyses/ana_invalid_str"), {
          finding: "No abnormal finding flagged",
          probability: 0.2,
          modelVersion: "1.0.0",
          source: "upload",
          createdAt: "not-a-timestamp",
        })
      );

      // Invalid type for createdAt (number)
      await assertFails(
        setDoc(doc(db, "users/doctor_uid/analyses/ana_num_created"), {
          finding: "No abnormal finding flagged",
          probability: 0.2,
          modelVersion: "1.0.0",
          source: "upload",
          createdAt: Date.now(),
        })
      );
    });

    // -------------------------------------------------------------------------
    // 8. Integration with Shared Web Payload Builders
    // -------------------------------------------------------------------------

    it("allows signup profile without institution using buildUserProfile", async () => {
      const db = getAuthedDb("user_no_inst", true);
      const payload = buildUserProfile({
        displayName: "Dr. Gregory House",
        role: "Doctor",
      });
      // Verify payload has no institution key
      if ("institution" in payload) {
        throw new Error("Expected buildUserProfile without institution to omit key");
      }
      await assertSucceeds(setDoc(doc(db, "users/user_no_inst"), payload));
    });

    it("allows signup profile with empty institution using buildUserProfile", async () => {
      const db = getAuthedDb("user_empty_inst", true);
      const payload = buildUserProfile({
        displayName: "Dr. James Wilson",
        role: "Doctor",
        institution: "",
      });
      await assertSucceeds(setDoc(doc(db, "users/user_empty_inst"), payload));
    });

    it("allows signup profile with institution using buildUserProfile", async () => {
      const db = getAuthedDb("user_with_inst", true);
      const payload = buildUserProfile({
        displayName: "Dr. Lisa Cuddy",
        role: "Doctor",
        institution: "Princeton-Plainsboro Teaching Hospital",
      });
      await assertSucceeds(setDoc(doc(db, "users/user_with_inst"), payload));
    });

    it("allows analysis from an upload using buildAnalysisRecord with default createdAt", async () => {
      const db = getAuthedDb("doc_upload_user", true);
      const payload = buildAnalysisRecord({
        finding: "Possible abnormal finding",
        probability: 0.874,
        modelVersion: "1.0.0",
        source: "upload",
      });
      assert.ok(payload.createdAt, "createdAt must be populated by builder default");
      await assertSucceeds(
        setDoc(doc(db, "users/doc_upload_user/analyses/rec_upload"), payload)
      );
    });

    it("allows analysis from a sample using buildAnalysisRecord with default createdAt", async () => {
      const db = getAuthedDb("doc_sample_user", true);
      const payload = buildAnalysisRecord({
        finding: "No abnormal finding flagged",
        probability: 0.142,
        modelVersion: "1.0.0",
        source: "sample",
      });
      assert.ok(payload.createdAt, "createdAt must be populated by builder default");
      await assertSucceeds(
        setDoc(doc(db, "users/doc_sample_user/analyses/rec_sample"), payload)
      );
    });

    it("rejects an analysis written with the old 'cluster' field", async () => {
      const db = getAuthedDb("doctor_uid", true);
      await assertFails(
        setDoc(doc(db, "users/doctor_uid/analyses/ana_with_cluster"), {
          finding: "Possible abnormal finding",
          probability: 0.85,
          modelVersion: "1.0.0",
          source: "upload",
          cluster: 2,
          createdAt: serverTimestamp(),
        })
      );
    });

    it("rejects an analysis written with the old 'label' field", async () => {
      const db = getAuthedDb("doctor_uid", true);
      await assertFails(
        setDoc(doc(db, "users/doctor_uid/analyses/ana_with_label"), {
          finding: "Possible abnormal finding",
          probability: 0.85,
          modelVersion: "1.0.0",
          source: "upload",
          label: "Abnormal",
          createdAt: serverTimestamp(),
        })
      );
    });
  });
});
