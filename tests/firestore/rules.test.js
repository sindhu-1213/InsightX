/**
 * Firestore Security Rules Unit Tests using @firebase/rules-unit-testing.
 *
 * Verifies:
 * - Only signed-in users with a verified email can read/write their own data.
 * - Complete denial of access to other users' data.
 * - Strict schema validation on users/{uid} (displayName, role in [Doctor, Staff, Student], institution < 100 chars).
 * - Strict schema validation on analyses/{id} (label, probability in [0, 1], cluster, modelVersion, createdAt).
 * - Immutability: analyses documents cannot be updated after creation.
 */

const {
  initializeTestEnvironment,
  assertFails,
  assertSucceeds,
} = require("@firebase/rules-unit-testing");
const { doc, getDoc, setDoc, updateDoc } = require("firebase/firestore");
const fs = require("fs");
const path = require("path");

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
          label: "Normal",
          probability: 0.0821,
          cluster: 0,
          modelVersion: "1.0.0",
          createdAt: new Date().toISOString(),
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
          label: "Abnormal",
          probability: 0.9412,
          cluster: 1,
          modelVersion: "1.0.0",
          createdAt: new Date().toISOString(),
        })
      );

      // 2. Update is strictly denied
      await assertFails(
        updateDoc(analysisRef, {
          label: "Normal",
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
          label: "Abnormal",
          probability: 1.5,
          cluster: 0,
          modelVersion: "1.0.0",
          createdAt: new Date().toISOString(),
        })
      );

      await assertFails(
        setDoc(refLow, {
          label: "Normal",
          probability: -0.1,
          cluster: 0,
          modelVersion: "1.0.0",
          createdAt: new Date().toISOString(),
        })
      );
    });

    it("denies invalid labels (must be Normal or Abnormal)", async () => {
      const db = getAuthedDb("doctor_uid", true);
      const analysisRef = doc(db, "users/doctor_uid/analyses/analysis_bad_label");

      await assertFails(
        setDoc(analysisRef, {
          label: "Pneumonia",
          probability: 0.85,
          cluster: 0,
          modelVersion: "1.0.0",
          createdAt: new Date().toISOString(),
        })
      );
    });

    it("denies extra fields in analyses documents", async () => {
      const db = getAuthedDb("doctor_uid", true);
      const analysisRef = doc(db, "users/doctor_uid/analyses/analysis_extra");

      await assertFails(
        setDoc(analysisRef, {
          label: "Normal",
          probability: 0.05,
          cluster: 0,
          modelVersion: "1.0.0",
          createdAt: new Date().toISOString(),
          diagnosisNotes: "Patient has cough",
        })
      );
    });

    it("denies cross-user access to analyses subcollection", async () => {
      const aliceDb = getAuthedDb("alice", true);
      const bobDb = getAuthedDb("bob", true);

      await assertSucceeds(
        setDoc(doc(aliceDb, "users/alice/analyses/ana_1"), {
          label: "Normal",
          probability: 0.05,
          cluster: 0,
          modelVersion: "1.0.0",
          createdAt: new Date().toISOString(),
        })
      );

      await assertFails(getDoc(doc(bobDb, "users/alice/analyses/ana_1")));
      await assertFails(
        setDoc(doc(bobDb, "users/alice/analyses/ana_1"), {
          label: "Abnormal",
          probability: 0.99,
          cluster: 1,
          modelVersion: "1.0.0",
          createdAt: new Date().toISOString(),
        })
      );
    });
  });
});
