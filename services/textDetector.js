/**
 * ZeroGPT-style AI text detector using multi-stage statistical analysis.
 *
 * Methodology (simulating DeepAnalyse™):
 *   1. Perplexity — how predictable/surprising word choices are (word frequency rarity)
 *   2. Burstiness — variance in sentence length and structure
 *   3. Vocabulary diversity — type-token ratio + hapax legomena rate
 *   4. Repetition — repeated phrase/n-gram frequency
 *   5. Sentence start diversity — how varied sentence beginnings are
 *   6. Ensemble classifier — weighted combination of all signals
 *   7. Sentence-level highlighting — per-sentence AI probability
 */

// Compact English word frequency map (log10 freq per million, from SUBTLEX-US)
// Words not in this list get a rarity score of 2.0 (very rare / human-like)
const WORD_FREQ = {
  the: 4.87, of: 4.63, and: 4.58, to: 4.51, a: 4.48, in: 4.42, is: 3.96,
  it: 3.92, you: 3.91, that: 3.90, he: 3.83, was: 3.80, for: 3.75, on: 3.73,
  are: 3.70, with: 3.68, as: 3.65, i: 3.63, his: 3.61, they: 3.57, be: 3.55,
  at: 3.54, one: 3.51, have: 3.49, this: 3.47, from: 3.46, or: 3.45, had: 3.43,
  by: 3.41, but: 3.40, not: 3.39, what: 3.37, all: 3.36, were: 3.35, we: 3.34,
  when: 3.33, can: 3.32, an: 3.31, who: 3.30, their: 3.29, there: 3.28, been: 3.27,
  if: 3.26, more: 3.25, will: 3.24, would: 3.23, her: 3.22, she: 3.21, do: 3.20,
  him: 3.19, has: 3.18, no: 3.17, which: 3.16, how: 3.15, its: 3.14, about: 3.13,
  out: 3.12, up: 3.11, them: 3.10, then: 3.09, so: 3.08, some: 3.07, into: 3.06,
  could: 3.05, other: 3.04, than: 3.03, also: 3.02, time: 3.01, my: 3.00, two: 2.99,
  just: 2.98, over: 2.97, people: 2.96, first: 2.95, like: 2.94, very: 2.93,
  your: 2.92, new: 2.91, after: 2.90, between: 2.89, through: 2.88, where: 2.87,
  should: 2.86, well: 2.85, most: 2.84, here: 2.83, much: 2.82, many: 2.81,
  may: 2.80, even: 2.79, every: 2.78, such: 2.77, because: 2.76, made: 2.75,
  these: 2.74, did: 2.73, down: 2.72, way: 2.71, our: 2.70, while: 2.69,
  now: 2.68, each: 2.67, any: 2.66, before: 2.65, know: 2.64, too: 2.63,
  within: 2.62, government: 2.61, only: 2.60, year: 2.59, state: 2.58,
  world: 2.57, still: 2.56, great: 2.55, high: 2.54, old: 2.53, against: 2.52,
  own: 2.51, under: 2.50, back: 2.49, say: 2.48, work: 2.47, part: 2.46,
  life: 2.45, must: 2.44, right: 2.43, thing: 2.42, during: 2.41, take: 2.40,
  three: 2.39, long: 2.38, hand: 2.37, place: 2.36, small: 2.35, number: 2.34,
  however: 2.33, system: 2.32, show: 2.31, water: 2.30, large: 2.29, early: 2.28,
  need: 2.27, find: 2.26, end: 2.25, another: 2.24, always: 2.23, good: 2.22,
  get: 2.21, make: 2.20, look: 2.19, head: 2.18, use: 2.17, since: 2.16,
  group: 2.15, point: 2.14, child: 2.13, city: 2.12, become: 2.11, public: 2.10,
  different: 2.09, include: 2.08, play: 2.06, present: 2.05,
  follow: 2.04, both: 2.03, house: 2.02, general: 2.01, again: 2.00,
  // Additional common words with lower frequencies
  important: 1.99, research: 1.98, development: 1.97, technology: 1.96,
  information: 1.95, education: 1.94, national: 1.93, economic: 1.92,
  political: 1.91, social: 1.90, process: 1.89, provide: 1.88, result: 1.87,
  require: 1.86, community: 1.85, support: 1.84, specific: 1.83, individual: 1.82,
  rather: 1.81, according: 1.80, analysis: 1.79, approach: 1.78, area: 1.77,
  available: 1.76, benefit: 1.75, cause: 1.74, change: 1.73, condition: 1.72,
  consider: 1.71, create: 1.70, data: 1.69, decision: 1.68, describe: 1.67,
  effect: 1.66, establish: 1.64, evidence: 1.63, experience: 1.62,
  factor: 1.61, focus: 1.60, force: 1.59, function: 1.58, growth: 1.57,
  identify: 1.56, impact: 1.55, increase: 1.54, issue: 1.53, language: 1.52,
  level: 1.51, maintain: 1.50, measure: 1.49, model: 1.48, movement: 1.47,
  necessary: 1.46, occur: 1.45, operation: 1.44, organization: 1.43, period: 1.42,
  policy: 1.41, population: 1.40, position: 1.39, potential: 1.38,
  previous: 1.36, primary: 1.35, produce: 1.34, professor: 1.33, program: 1.32,
  project: 1.31, property: 1.30, propose: 1.29, range: 1.28, rate: 1.27,
  recent: 1.26, reduce: 1.25, region: 1.24, relate: 1.23, relationship: 1.22,
  resource: 1.21, response: 1.20, role: 1.19, section: 1.18, security: 1.17,
  service: 1.16, similar: 1.14, source: 1.13, standard: 1.12,
  strategy: 1.11, structure: 1.10, study: 1.09, subject: 1.08, success: 1.07,
  suggest: 1.06, theory: 1.05,   traditional: 1.04, training: 1.03, value: 1.02,
  activity: 1.00, address: 0.99, agreement: 0.98, allow: 0.97,
  application: 0.96, article: 0.95, attention: 0.94, authority: 0.93, basis: 0.92,
  category: 0.91, challenge: 0.90, claim: 0.89, collection: 0.88, combination: 0.87,
  communication: 0.86, comparison: 0.85, complex: 0.84, component: 0.83,
  conclusion: 0.82, conference: 0.81, conflict: 0.80, connection: 0.79,
  consequence: 0.78, construction: 0.77, consumer: 0.76, context: 0.75,
  contribution: 0.74, control: 0.73, controversy: 0.72, convention: 0.71,
  correlation: 0.70, crisis: 0.69, criteria: 0.68, critical: 0.67, culture: 0.66,
  debate: 0.65, decade: 0.64, decline: 0.63, define: 0.62, definition: 0.61,
  delivery: 0.60, demand: 0.59, demonstrate: 0.58, department: 0.57, depend: 0.56,
  design: 0.55, despite: 0.54, destination: 0.53, destruction: 0.52, detect: 0.51,
  determine: 0.50, digital: 0.49, dimension: 0.48, direction: 0.47, discipline: 0.46,
  discovery: 0.45, discrimination: 0.44, discussion: 0.43, disease: 0.42,
  display: 0.41, distinction: 0.40, distribution: 0.39, diversity: 0.38,
  document: 0.37, domain: 0.36, domestic: 0.35, dominance: 0.34, draft: 0.33,
  dramatic: 0.32, duration: 0.31, dynamic: 0.30, economy: 0.29, edition: 0.28,
  element: 0.27, elimination: 0.26, emergence: 0.25, emotion: 0.24, emphasis: 0.23,
  empire: 0.22, employee: 0.21, enable: 0.20, encounter: 0.19, enforcement: 0.18,
  engagement: 0.17, enormous: 0.16, enterprise: 0.15, entertainment: 0.14,
  enthusiasm: 0.13, entire: 0.12, entity: 0.11, entrepreneur: 0.10,
  entry: 0.09, environment: 0.08, episode: 0.07, equality: 0.06, equation: 0.05,
  equipment: 0.04, equivalent: 0.03, error: 0.02, especially: 0.01, essence: 0.00,
  // Deliberately add some "AI-favored" transition words at higher freqs
  furthermore: 1.50, moreover: 1.48, additionally: 1.45, consequently: 1.42,
  nevertheless: 1.40, subsequently: 1.38, importantly: 1.35, notably: 1.33,
  particularly: 1.30, significantly: 1.28, therefore: 1.25, thus: 1.23,
  indeed: 1.20, overall: 1.18, typically: 1.15, ultimately: 1.13,
  comprehensive: 1.10, extensive: 1.08, substantial: 1.05, numerous: 1.03,
  diverse: 1.00, various: 0.98, multiple: 0.96, robust: 0.95, crucial: 0.94,
  essential: 0.93, fundamental: 0.92, significant: 0.91, innovative: 0.90,
  emerging: 0.89, evolving: 0.88, sophisticated: 0.87, advanced: 0.86,
  cutting: 0.85, groundbreaking: 0.84, unprecedented: 0.83, transformative: 0.82,
  revolutionize: 0.81, paradigm: 0.80, landscape: 0.79, realm: 0.78,
  tapestry: 0.77, navigate: 0.76, delve: 0.75, explore: 0.74, foster: 0.73,
  enhance: 0.72, leverage: 0.71, optimize: 0.70, streamline: 0.69,
  facilitate: 0.68, implement: 0.67, utilize: 0.66, prioritize: 0.65,
  synergize: 0.64, ecosystem: 0.63, stakeholder: 0.62, initiative: 0.61,
  framework: 0.60, methodology: 0.59, best: 0.58, practice: 0.57,
};

function tokenize(text) {
  return text.toLowerCase()
    .replace(/['']/g, '')     // normalize apostrophes
    .replace(/[^a-z0-9\s-]/g, ' ')
    .split(/\s+/)
    .filter(w => w.length > 0 && /[a-z]/.test(w));
}

// Colloquial/casual words that humans use but AI avoids
const COLLOQUIAL = new Set([
  'guy', 'guys', 'dude', 'buddy', 'pal', 'folks', 'stuff', 'thing', 'things',
  'kinda', 'sorta', 'pretty', 'quite', 'actually', 'basically', 'honestly',
  'anyway', 'anyways', 'oh', 'ah', 'wow', 'hey', 'yeah', 'yep', 'nope', 'nah',
  'cool', 'awesome', 'amazing', 'nice', 'sucks', 'awful', 'terrible', 'horrible',
  'super', 'totally', 'absolutely', 'definitely', 'maybe', 'perhaps',
  'gonna', 'wanna', 'gotta', 'coulda', 'shoulda', 'woulda', 'gimme', 'lemme',
  'cuz', 'cause', 'cos', 'til', 'till', 'ok', 'okay', 'alright', 'cmon',
  'c\'mon', 'omg', 'lol', 'yea', 'nah', 'dunno', 'prolly', 'probly',
  'love', 'hate', 'feel', 'felt', 'guess', 'think', 'thought', 'wonder',
  'hopefully', 'thankfully', 'luckily', 'sadly', 'funny', 'weird', 'strange',
  'lovely', 'horrible', 'brilliant', 'ridiculous', 'silly', 'dumb', 'smart',
  'busy', 'tired', 'hungry', 'thirsty', 'bored', 'excited', 'scared',
  'happy', 'sad', 'mad', 'glad', 'sorry', 'sure', 'fine', 'good', 'bad',
  'better', 'worse', 'best', 'worst', 'easy', 'hard', 'simple', 'quick',
  'fast', 'slow', 'big', 'small', 'lots', 'loads', 'heaps', 'tons',
  'bit', 'little', 'much', 'many', 'few', 'couple', 'bunch', 'buncha',
  'everyone', 'everybody', 'nobody', 'someone', 'somebody', 'anyone', 'anybody',
  'now', 'today', 'yesterday', 'tomorrow', 'morning', 'afternoon', 'evening',
  'night', 'week', 'month', 'year', 'ago', 'later', 'earlier', 'soon',
  'here', 'there', 'everywhere', 'somewhere', 'anywhere', 'nowhere',
  'always', 'never', 'sometimes', 'often', 'usually', 'rarely', 'seldom',
  'already', 'still', 'yet', 'anymore', 'again', 'once', 'twice',
]);

// AI-favored transition/formality words (overused by LLMs)
const AI_TRANSITIONS = new Set([
  'furthermore', 'moreover', 'additionally', 'consequently', 'nevertheless',
  'notwithstanding', 'subsequently', 'therefore', 'thus', 'hence', 'indeed',
  'overall', 'ultimately', 'importantly', 'notably', 'significantly',
  'conversely', 'meanwhile', 'accordingly', 'further', 'likewise',
  'moreover', 'thereafter', 'thereby', 'therein', 'thereupon',
  'comprehensive', 'extensive', 'substantial', 'numerous', 'diverse',
  'robust', 'crucial', 'essential', 'fundamental', 'significant',
  'innovative', 'emerging', 'evolving', 'sophisticated', 'advanced',
  'cutting-edge', 'groundbreaking', 'unprecedented', 'transformative',
  'leverage', 'leverages', 'leveraging', 'facilitate', 'facilitates',
  'facilitating', 'implement', 'implements', 'implementing', 'implementation',
  'utilize', 'utilizes', 'utilizing', 'utilization', 'prioritize',
  'prioritizes', 'prioritizing', 'optimize', 'optimizes', 'optimizing',
  'streamline', 'streamlines', 'streamlining', 'revolutionize',
  'revolutionizes', 'revolutionizing', 'paradigm', 'landscape', 'realm',
  'tapestry', 'navigate', 'navigates', 'navigating', 'delve', 'delves',
  'delving', 'foster', 'fosters', 'fostering', 'enhance', 'enhances',
  'enhancing', 'showcase', 'showcases', 'showcasing', 'underscore',
  'underscores', 'underscoring', 'highlight', 'highlights', 'highlighting',
  'demonstrate', 'demonstrates', 'demonstrating', 'showcasing',
  'ecosystem', 'stakeholder', 'stakeholders', 'initiative', 'initiatives',
  'framework', 'frameworks', 'methodology', 'methodologies',
  'bespoke', 'holistic', 'seamless', 'seamlessly', 'meaningful',
  'actionable', 'scalable', 'tailored', 'dynamic', 'synergy', 'synergize',
  'synergistic', 'mission-critical', 'core', 'core', 'best-in-class',
  'best-of-breed', 'cutting-edge', 'state-of-the-art', 'next-generation', 
  'game-changer', 'game-changing', 'industry-leading',
  'navigate', 'navigating', 'complexities', 'intricate', 'intricacies',
  'nuanced', 'nuance', 'nuances', 'multifaceted', 'multifarious',
  'plethora', 'multitude', 'multitude', 'myriad', 'array', 'wide array',
  'delve', 'delving', 'underscore', 'underscores', 'underscoring',
]);

function hasColloquial(words) {
  return words.some(w => COLLOQUIAL.has(w));
}

function countAiTransitions(words) {
  return words.filter(w => AI_TRANSITIONS.has(w)).length;
}

function splitSentences(text) {
  // Split on sentence-ending punctuation, handling common abbreviations
  const raw = text.split(/(?<=[.!?])\s+/);
  return raw.filter(s => s.trim().length > 0);
}

function getWordRarity(word) {
  const freq = WORD_FREQ[word];
  if (freq !== undefined) {
    return Math.max(0, 4.87 - freq) / 4.87; // 0 = very common, 1 = very rare among known words
  }
  return 1.0; // Unknown words = rare (human-like)
}

/**
 * Perplexity score: how predictable/safe are the word choices.
 * AI text uses high-frequency common words and avoids surprising vocabulary.
 * Human text uses more varied, domain-specific, and low-frequency words.
 * Returns 0-100 where higher = more AI-like (low perplexity / safe choices).
 */
function calculatePerplexity(words) {
  if (words.length < 3) return 50;

  // Average word frequency rank (higher freq = more common = more AI-like)
  const top500 = new Set([...Object.keys(WORD_FREQ)].slice(0, 500));
  const knownWords = words.filter(w => WORD_FREQ[w] !== undefined);
  const unknownWords = words.filter(w => WORD_FREQ[w] === undefined);

  // Ratio of words in top 500 most common
  const top500Ratio = knownWords.length > 0
    ? words.filter(w => top500.has(w)).length / words.length
    : 0;

  // Ratio of unknown/rare words (not in our frequency list at all)
  const unknownRatio = unknownWords.length / words.length;

  // Average frequency of known words (higher = more common = AI-like)
  const avgFreq = knownWords.length > 0
    ? knownWords.reduce((sum, w) => sum + WORD_FREQ[w], 0) / knownWords.length
    : 0;

  // Normalize avgFreq (range ~0-4.87) to 0-100
  const freqScore = Math.min(100, (avgFreq / 3.0) * 100);

  // Low unknown ratio = AI-like (AI avoids rare/specialized words)
  // High top500 ratio = AI-like (AI sticks to common words)
  const rarePenalty = Math.min(100, unknownRatio * 200);
  const commonBoost = Math.min(100, top500Ratio * 120);

  return Math.round(freqScore * 0.35 + commonBoost * 0.35 + (100 - rarePenalty) * 0.30);
}

/**
 * Formality score: measures ratio of formal AI-like words vs casual human words.
 * AI text avoids colloquial language and overuses formal transitions.
 * Returns 0-100 where higher = more AI-like (formal, no colloquialisms).
 */
function calculateFormality(words) {
  if (words.length < 5) return 50;
  const colloquialCount = words.filter(w => COLLOQUIAL.has(w)).length;
  const transitionCount = countAiTransitions(words);

  // Low colloquial density = AI-like (AI avoids casual language)
  const colloquialRatio = colloquialCount / words.length;
  const colloquialScore = colloquialRatio < 0.02 ? 70 : colloquialRatio < 0.06 ? 40 : 15;

  // High transition density = AI-like
  const transitionRatio = transitionCount / words.length;
  const transitionScore = transitionRatio > 0.08 ? 90 : transitionRatio > 0.04 ? 65 : transitionRatio > 0.01 ? 35 : 10;

  return Math.round(colloquialScore * 0.5 + transitionScore * 0.5);
}

/**
 * Burstiness score: variance in sentence length and structure.
 * Low burstiness (AI-like): uniform sentence lengths.
 * High burstiness (human-like): varied sentence lengths.
 * Returns 0-100 where higher = more AI-like.
 */
function calculateBurstiness(sentences) {
  if (sentences.length < 3) return 50;
  const lengths = sentences.map(s => s.split(/\s+/).length);
  const avg = lengths.reduce((a, b) => a + b, 0) / lengths.length;
  const variance = lengths.reduce((sum, l) => sum + (l - avg) ** 2, 0) / lengths.length;
  const stdDev = Math.sqrt(variance);

  // Humans have higher std dev relative to mean (coefficient of variation)
  const cv = avg > 0 ? stdDev / avg : 0;

  // Low CV (~0.3-0.5) = AI-like, High CV (~0.8-2.0) = human-like
  // Map: cv < 0.5 → 80-100 (AI-like), cv > 1.0 → 0-20 (human-like)
  let score;
  if (cv < 0.3) score = 95;
  else if (cv < 0.5) score = 80 + (0.5 - cv) * 75;
  else if (cv < 0.7) score = 50 + (0.7 - cv) * 150;
  else if (cv < 1.0) score = 20 + (1.0 - cv) * 100;
  else score = Math.max(0, 20 - (cv - 1.0) * 30);

  return Math.round(Math.min(100, Math.max(0, score)));
}

/**
 * Vocabulary diversity score: lexical richness and word frequency profile.
 * AI text tends to use a narrower band of mid-to-high-frequency words.
 * Human text uses more low-frequency, domain-specific, and varied words.
 * Returns 0-100 where higher = more AI-like.
 */
function calculateVocabularyDiversity(words) {
  if (words.length < 5) return 50;

  const freqMap = new Map();
  words.forEach(w => freqMap.set(w, (freqMap.get(w) || 0) + 1));

  // Ratio of words that are LOW frequency (not in top 500)
  const top500 = new Set([...Object.keys(WORD_FREQ)].slice(0, 500));
  const lowFreqRatio = words.filter(w => !top500.has(w)).length / words.length;
  // Higher lowFreqRatio = more rare/specific words = human-like
  const lowFreqScore = Math.min(100, (1 - lowFreqRatio) * 130);

  // Average repetition of the most common word
  const maxFreq = Math.max(...freqMap.values());
  const maxProp = maxFreq / words.length;
  // High maxProp = one word dominates = AI-like
  const maxScore = maxProp > 0.10 ? 80 : maxProp > 0.06 ? 50 : 20;

  // Lexical diversity: how many different words relative to length
  // For short texts (<100 words), use a simpler scale
  const ttr = freqMap.size / words.length;
  let ttrScore;
  if (words.length < 30) {
    // Short text: TTR > 0.85 is expected, < 0.65 is suspicious
    ttrScore = ttr < 0.65 ? 70 : ttr < 0.80 ? 40 : 20;
  } else {
    // Longer text: TTR < 0.45 is AI-like, > 0.60 is human-like
    ttrScore = ttr < 0.35 ? 90 : ttr < 0.45 ? 65 : ttr < 0.60 ? 40 : 20;
  }

  return Math.round(lowFreqScore * 0.5 + maxScore * 0.25 + ttrScore * 0.25);
}

/**
 * Repetition score: frequency of repeated words, bigrams, and structural patterns.
 * AI text tends to repeat the same sentence structures, transition words, and framing.
 * Human text has more varied expression and less predictable patterns.
 * Returns 0-100 where higher = more AI-like.
 */
function calculateRepetition(words) {
  if (words.length < 10) return 50;

  // Count repeated bigrams
  const bigrams = new Map();
  for (let i = 0; i < words.length - 1; i++) {
    const gram = words[i] + ' ' + words[i + 1];
    bigrams.set(gram, (bigrams.get(gram) || 0) + 1);
  }
  const totalBigrams = Math.max(1, words.length - 1);
  const repeatedBigrams = [...bigrams.values()].filter(c => c > 1).length;
  const bigramRepRatio = repeatedBigrams / totalBigrams;

  // Count how many function/transition words repeat
  const transitions = ['furthermore', 'moreover', 'additionally', 'consequently',
    'nevertheless', 'subsequently', 'therefore', 'thus', 'indeed', 'overall',
    'typically', 'ultimately', 'importantly', 'notably', 'significantly',
    'in', 'to', 'the', 'a', 'an', 'and', 'or', 'but', 'of', 'for', 'with'];
  const transCount = words.filter(w => transitions.includes(w)).length;
  const transRatio = transCount / words.length;

  // Repetition via stop-word density (high = generic = AI-like)
  const stopwords = new Set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
    'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'can', 'could', 'shall', 'should', 'may', 'might', 'must', 'it', 'its',
    'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they']);
  const stopwordRatio = words.filter(w => stopwords.has(w)).length / words.length;
  // stopwordRatio > 0.55 = very generic = AI-like

  const bigramScore = Math.min(100, bigramRepRatio * 500);
  const transScore = Math.min(100, (transRatio / 0.15) * 100);
  const stopScore = stopwordRatio > 0.55
    ? Math.min(100, (stopwordRatio - 0.40) * 200)
    : Math.max(0, (stopwordRatio - 0.25) * 200);

  return Math.round(bigramScore * 0.25 + transScore * 0.45 + stopScore * 0.30);
}

/**
 * Punctuation and structure diversity.
 * Humans use more varied punctuation (—, ..., !, ?, parentheses, quotes).
 * AI text tends to use standard punctuation only (. ,).
 * Returns 0-100 where higher = more AI-like.
 */
function calculatePunctuationDiversity(text) {
  if (text.length < 30) return 50;
  const exclamations = (text.match(/!/g) || []).length;
  const questions = (text.match(/\?/g) || []).length;
  const dashes = (text.match(/—/g) || []).length;
  const ellipsis = (text.match(/\.\.\./g) || []).length;
  const quotes = (text.match(/["""]/g) || []).length;
  const parens = (text.match(/[()]/g) || []).length;

  const totalVaried = exclamations + questions + dashes + ellipsis + quotes + parens;
  const textLen = text.length;

  // Less than 1 varied punctuation per 200 chars = AI-like (too uniform)
  const density = totalVaried / textLen;

  let score;
  if (density < 0.005) score = 70;
  else if (density < 0.015) score = 45;
  else if (density < 0.03) score = 25;
  else score = 10;

  return score;
}

/**
 * Sentence start diversity: how varied sentence beginnings are.
 * AI text favors formal transitions (Furthermore, Additionally, However, Thus)
 * and often repeats the same structural patterns.
 * Human text has more varied, natural openings.
 * Returns 0-100 where higher = more AI-like.
 */
function calculateSentenceStartDiversity(sentences) {
  if (sentences.length < 2) return 50;
  const starts = sentences.map(s => {
    const first = tokenize(s)[0];
    return first || '';
  }).filter(s => s.length > 0);

  if (starts.length < 2) return 50;

  const unique = new Set(starts);
  const ratio = unique.size / starts.length;

  // Check for AI-favored transition starters
  const aiStarters = ['furthermore', 'moreover', 'additionally', 'consequently',
    'nevertheless', 'subsequently', 'therefore', 'thus', 'indeed', 'overall',
    'ultimately', 'importantly', 'notably', 'however', 'in', 'the', 'this',
    'these', 'it', 'additionally', 'conversely', 'meanwhile', 'accordingly'];
  const aiStarterCount = starts.filter(w => aiStarters.includes(w)).length;
  const aiStarterRatio = aiStarterCount / starts.length;

  // Low diversity + high AI starter ratio = AI-like
  let diversityScore;
  if (ratio < 0.4) diversityScore = 80;
  else if (ratio < 0.6) diversityScore = 50 + (0.6 - ratio) * 150;
  else if (ratio < 0.8) diversityScore = 20 + (0.8 - ratio) * 100;
  else diversityScore = Math.max(5, 20 - (ratio - 0.8) * 75);

  const starterScore = Math.min(100, aiStarterRatio * 150);

  return Math.round(diversityScore * 0.5 + starterScore * 0.5);
}

/**
 * Highlight sentences that are most likely AI-generated.
 * Returns array of { sentence, score, isAi } objects.
 */
function highlightSentences(sentences, globalMetrics) {
  if (sentences.length === 0) return [];
  return sentences.map(sentence => {
    const trimmed = sentence.trim();
    if (trimmed.length < 10) return { sentence: trimmed, score: 50, isAi: false };
    const sentWords = tokenize(trimmed);
    if (sentWords.length < 3) return { sentence: trimmed, score: 50, isAi: false };
    const sentSplits = splitSentences(trimmed);
    const perplexity = calculatePerplexity(sentWords);
    const burstiness = calculateBurstiness(sentSplits);
    const vocab = calculateVocabularyDiversity(sentWords);
    const repetition = calculateRepetition(sentWords);
    const formality = calculateFormality(sentWords);
    const score = Math.round(
      perplexity * 0.15 + burstiness * 0.12 + vocab * 0.13 + repetition * 0.15 + formality * 0.25 +
      ((globalMetrics?.sentenceStarts ?? 50) * 0.10) + ((globalMetrics?.punctuation ?? 50) * 0.10)
    );
    return { sentence: trimmed, score, isAi: score >= 55 };
  });
}

/**
 * Main entry point. Analyzes text and returns ZeroGPT-style results.
 *
 * @param {string} text - The text to analyze
 * @returns {object} { overallScore, perplexity, burstiness, vocabulary, repetition, sentenceStarts, sentences }
 */
export function analyzeText(text) {
  if (!text || typeof text !== 'string' || text.trim().length === 0) {
    return {
      overallScore: 0,
      perplexity: 0,
      burstiness: 0,
      vocabulary: 0,
      repetition: 0,
      sentenceStarts: 0,
      sentences: [],
    };
  }

  const cleaned = text.trim();
  const words = tokenize(cleaned);
  const sentences = splitSentences(cleaned);

  if (words.length < 3) {
    return { overallScore: 0, perplexity: 0, burstiness: 0, vocabulary: 0, repetition: 0, sentenceStarts: 0, sentences: [] };
  }

  const perplexity = calculatePerplexity(words);
  const burstiness = calculateBurstiness(sentences);
  const vocabulary = calculateVocabularyDiversity(words);
  const repetition = calculateRepetition(words);
  const sentenceStarts = calculateSentenceStartDiversity(sentences);
  const formality = calculateFormality(words);
  const punctuation = calculatePunctuationDiversity(cleaned);
  const sentencesHighlight = highlightSentences(sentences, {
    perplexity, burstiness, vocabulary, repetition, sentenceStarts, formality, punctuation,
    words, cleaned,
  });

  // Apply text length confidence adjustment
  let shortTextPenalty = 1.0;
  if (words.length < 10) shortTextPenalty = 0.3;
  else if (words.length < 20) shortTextPenalty = 0.5;
  else if (words.length < 40) shortTextPenalty = 0.75;

  // Ensemble: weighted combination (ZeroGPT-style multi-stage)
  // Formality and punctuation are strong signals; burstiness weak for short texts
  const burstWeight = sentences.length < 4 ? 0.10 : 0.20;
  const totalWeight = 0.15 + burstWeight + 0.15 + 0.15 + 0.15 + 0.25 + 0.10;
  const overallScore = Math.round(shortTextPenalty * (
    perplexity * 0.15 +
    burstiness * burstWeight +
    vocabulary * 0.15 +
    repetition * 0.15 +
    sentenceStarts * 0.15 +
    formality * 0.25 +
    punctuation * 0.10
  ) / totalWeight);

  const aiCount = sentencesHighlight.filter(s => s.isAi).length;
  const aiPercentage = sentencesHighlight.length > 0
    ? Math.round((aiCount / sentencesHighlight.length) * 100)
    : 0;

  return {
    overallScore: Math.min(100, Math.max(0, overallScore)),
    perplexity,
    burstiness,
    vocabulary,
    repetition,
    sentenceStarts,
    formality,
    punctuation,
    sentences: sentencesHighlight,
    aiSentencePercentage: aiPercentage,
  };
}
