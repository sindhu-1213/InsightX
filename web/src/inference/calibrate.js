/**
 * Platt scaling probability calibration and clinical decision logic.
 * Exact mathematical port of api/main.py postprocessing.
 */

export const RESEARCH_DISCLAIMER =
  'RESEARCH DEMO ONLY - This tool is not a certified medical device. ' +
  'Results are AI predictions/findings for research purposes only and must not be used to guide clinical decisions. ' +
  'Always consult a qualified radiologist.';

/**
 * Apply Platt scaling and generate decision metrics from raw model logit.
 *
 * @param {number} rawLogit Raw linear logit from ONNX model
 * @param {{ coef: number, intercept: number, threshold_youden: number }} calibConfig
 * @returns {object} Calibrated probabilities and decision labels
 */
export function calibratePrediction(rawLogit, calibConfig) {
  const coef = Number(calibConfig.coef);
  const intercept = Number(calibConfig.intercept);
  const threshold = Number(calibConfig.threshold_youden);

  const uncalibratedProb = 1.0 / (1.0 + Math.exp(-rawLogit));
  const calibratedLogit = coef * rawLogit + intercept;
  let calibratedProb = 1.0 / (1.0 + Math.exp(-calibratedLogit));
  calibratedProb = Math.min(1.0, Math.max(0.0, calibratedProb));

  const predCode = calibratedProb >= threshold ? 1 : 0;
  const predLabel = predCode === 1 ? 'Abnormal' : 'Normal';
  const findingLabel =
    predCode === 1 ? 'Possible abnormal finding' : 'No abnormal finding flagged';

  const limitationsText =
    'Research demonstration only. Not a medical device, and not intended for clinical use. ' +
    `At operating threshold (${threshold.toFixed(4)}), test sensitivity is approximately ` +
    '56.45%, meaning the model misses a significant proportion of abnormal cases.';

  return {
    label: predLabel,
    finding_label: findingLabel,
    threshold_label: findingLabel,
    finding: findingLabel,
    worded_label: findingLabel,
    prediction_code: predCode,
    calibrated_probability: Number(calibratedProb.toFixed(4)),
    uncalibrated_probability: Number(uncalibratedProb.toFixed(4)),
    threshold_used: Number(threshold.toFixed(4)),
    limitations: limitationsText,
    projection_hint: 'Frontal chest X-ray expected (PA or AP view recommended).',
    disclaimer: RESEARCH_DISCLAIMER
  };
}

