export const PLAYER_CLUBS = {
    'neymar':       ['santos', 'barcelona', 'psg', 'paris saint-germain', 'paris sg', 'al-hilal', 'al hilal'],
    'ronaldinho':   ['gremio', 'psg', 'paris saint-germain', 'barcelona', 'ac milan', 'flamengo', 'atletico mineiro', 'fluminense', 'querétaro', 'atletico asuncion'],
    'roberto carlos': ['corinthians', 'palmeiras', 'inter de limeira', 'real madrid', 'fenerbahce', 'corinthians', 'delhi dynamos'],
    'cafu':         ['juventude', 'santos', 'zaragoza', 'juventus', 'roma', 'ac milan'],
    'lucas paqueta': ['flamengo', 'ac milan', 'lyon', 'west ham'],
    'vinicius':     ['flamengo', 'real madrid'],
    'vinicius junior': ['flamengo', 'real madrid'],
    'rodrygo':      ['santos', 'real madrid'],
    'gabriel jesus': ['palmeiras', 'manchester city', 'arsenal'],
    'casemiro':     ['real madrid', 'manchester united'],
    'thiago silva': ['fluminense', 'ac milan', 'psg', 'paris saint-germain', 'chelsea'],
    'david luiz':   ['vasco', 'vitoria', 'benfica', 'chelsea', 'psg', 'paris saint-germain', 'arsenal', 'flamengo'],
    'marcelo':      ['fluminense', 'real madrid', 'olympiakos'],
    'ronaldo':      ['sporting', 'sporting cp', 'manchester united', 'real madrid', 'juventus', 'al nassr'],
    'cristiano ronaldo': ['sporting', 'sporting cp', 'manchester united', 'real madrid', 'juventus', 'al nassr'],
    'bruno fernandes': ['novara', 'udinese', 'sampdoria', 'sporting', 'sporting cp', 'manchester united'],
    'bernardo silva': ['benfica', 'monaco', 'manchester city'],
    'ruben dias':   ['benfica', 'manchester city'],
    'joao felix':   ['benfica', 'atletico madrid', 'chelsea', 'barcelona', 'atletico madrid'],
    'messi':        ['barcelona', 'psg', 'paris saint-germain', 'inter miami'],
    'lionel messi': ['barcelona', 'psg', 'paris saint-germain', 'inter miami'],
    'di maria':     ['rosario central', 'benfica', 'real madrid', 'manchester united', 'psg', 'paris saint-germain', 'juventus', 'benfica'],
    'higuain':      ['river plate', 'real madrid', 'napoli', 'juventus', 'ac milan', 'chelsea', 'inter miami'],
    'dybala':       ['palermo', 'juventus', 'roma'],
    'lautaro':      ['racing club', 'inter', 'inter milan'],
    'lautaro martinez': ['racing club', 'inter', 'inter milan'],
    'mbappe':       ['monaco', 'psg', 'paris saint-germain', 'paris sg', 'real madrid'],
    'kylian mbappe': ['monaco', 'psg', 'paris saint-germain', 'paris sg', 'real madrid'],
    'griezmann':    ['sociedad', 'real sociedad', 'atletico madrid', 'barcelona', 'atletico madrid'],
    'benzema':      ['lyon', 'real madrid', 'al ittihad'],
    'karim benzema': ['lyon', 'real madrid', 'al ittihad'],
    'pogba':        ['manchester united', 'juventus'],
    'paul pogba':   ['manchester united', 'juventus'],
    'kante':        ['caen', 'leicester', 'chelsea', 'al ittihad'],
    'salah':        ['basel', 'chelsea', 'fiorentina', 'roma', 'liverpool'],
    'mo salah':     ['basel', 'chelsea', 'fiorentina', 'roma', 'liverpool'],
    'mohamed salah': ['basel', 'chelsea', 'fiorentina', 'roma', 'liverpool'],
    'harry kane':   ['tottenham', 'spurs', 'bayern', 'bayern munich'],
    'raheem sterling': ['liverpool', 'manchester city', 'chelsea'],
    'trent alexander-arnold': ['liverpool'],
    'bellingham':   ['birmingham', 'dortmund', 'borussia dortmund', 'real madrid'],
    'jude bellingham': ['birmingham', 'dortmund', 'borussia dortmund', 'real madrid'],
    'lewandowski':  ['lech poznan', 'borussia dortmund', 'dortmund', 'bayern', 'bayern munich', 'barcelona'],
    'robert lewandowski': ['lech poznan', 'borussia dortmund', 'dortmund', 'bayern', 'bayern munich', 'barcelona'],
    'muller':       ['bayern', 'bayern munich'],
    'thomas muller': ['bayern', 'bayern munich'],
    'neuer':        ['schalke', 'schalke 04', 'bayern', 'bayern munich'],
    'manuel neuer': ['schalke', 'schalke 04', 'bayern', 'bayern munich'],
    'pedri':        ['las palmas', 'barcelona'],
    'frenkie de jong': ['barcelona', 'ajax'],
    'de jong': ['barcelona', 'ajax'],
    'michael olise': ['bayern munich', 'crystal palace', 'reading'],
    'olise': ['bayern munich', 'crystal palace', 'reading'],
    'yamal':        ['barcelona'],
    'lamine yamal': ['barcelona'],
    'gavi':         ['barcelona'],
    'busquets':     ['barcelona', 'inter miami'],
    'sergio busquets': ['barcelona', 'inter miami'],
    'xavi':         ['barcelona', 'al sadd'],
    'iniesta':      ['barcelona', 'vissel kobe'],
    'van dijk':     ['groningen', 'celtic', 'southampton', 'liverpool'],
    'virgil van dijk': ['groningen', 'celtic', 'southampton', 'liverpool'],
    'de bruyne':    ['genk', 'werder bremen', 'chelsea', 'wolfsburg', 'manchester city'],
    'kevin de bruyne': ['genk', 'werder bremen', 'chelsea', 'wolfsburg', 'manchester city'],
    'totti':        ['roma'],
    'del piero':    ['juventus', 'sydney fc', 'delhi dynamos'],
    'pirlo':        ['brescia', 'inter', 'inter milan', 'ac milan', 'juventus', 'new york city', 'anderlecht'],
    'buffon':       ['parma', 'juventus', 'psg', 'paris saint-germain'],
    'baggio':       ['vicenza', 'fiorentina', 'juventus', 'ac milan', 'bologna', 'inter', 'brescia'],
};

/**
 * Celebrity contexts for deepfake mismatch detection.
 * Maps known names (matching face_api.py naming: lowercase, underscore-separated) to their
 * known affiliations, roles, and typical settings. When a Face API match returns a celebrity,
 * these contexts help identify suspicious out-of-context depictions.
 */
export const CELEBRITY_CONTEXTS = {
    'elon musk': {
        display: 'Elon Musk',
        roles: ['CEO', 'businessman', 'entrepreneur', 'tech executive'],
        organizations: ['Tesla', 'SpaceX', 'X', 'Twitter', 'Neuralink', 'The Boring Company', 'PayPal'],
        typicalSettings: ['tech conference', 'factory', 'press conference', 'office', 'Tesla event', 'SpaceX launch'],
        sport: null,
        party: null,
        opponents: [],
    },
    'taylor swift': {
        display: 'Taylor Swift',
        roles: ['singer', 'songwriter', 'musician', 'pop star', 'performer'],
        organizations: [],
        typicalSettings: ['concert', 'stage', 'music awards', 'recording studio', 'red carpet', 'music video set'],
        sport: null,
        party: null,
        opponents: [],
    },
    'donald trump': {
        display: 'Donald Trump',
        roles: ['president', 'politician', 'businessman', 'former president'],
        organizations: ['White House', 'Trump Organization', 'Republican Party'],
        typicalSettings: ['political rally', 'press conference', 'White House', 'debate stage', 'political event', 'inauguration', 'campaign event'],
        sport: null,
        party: 'Republican',
        opponents: ['Democratic Party', 'Joe Biden', 'Kamala Harris'],
    },
    'tom cruise': {
        display: 'Tom Cruise',
        roles: ['actor', 'film producer', 'movie star'],
        organizations: [],
        typicalSettings: ['movie premiere', 'red carpet', 'film set', 'awards ceremony', 'interview'],
        sport: null,
        party: null,
        opponents: [],
    },
    'joe biden': {
        display: 'Joe Biden',
        roles: ['president', 'politician', 'former president', 'vice president'],
        organizations: ['White House', 'Democratic Party', 'US Senate'],
        typicalSettings: ['political rally', 'press conference', 'White House', 'political event', 'state dinner', 'inauguration'],
        sport: null,
        party: 'Democratic',
        opponents: ['Republican Party', 'Donald Trump'],
    },
    'barack obama': {
        display: 'Barack Obama',
        roles: ['president', 'politician', 'former president', 'author'],
        organizations: ['White House', 'Democratic Party', 'Obama Foundation'],
        typicalSettings: ['political rally', 'press conference', 'speech', 'White House', 'political event', 'book signing'],
        sport: null,
        party: 'Democratic',
        opponents: ['Republican Party', 'Donald Trump', 'John McCain', 'Mitt Romney'],
    },
    'the rock': {
        display: 'Dwayne "The Rock" Johnson',
        roles: ['actor', 'wrestler', 'performer', 'entertainer'],
        organizations: ['WWE', 'United Football League', 'Seven Bucks Productions'],
        typicalSettings: ['movie premiere', 'wrestling ring', 'red carpet', 'film set', 'gym', 'press conference'],
        sport: 'wrestling',
        party: null,
        opponents: [],
    },
    'putin': {
        display: 'Vladimir Putin',
        roles: ['president', 'politician', 'world leader', 'former KGB officer'],
        organizations: ['Kremlin', 'Russian Government', 'United Russia'],
        typicalSettings: ['press conference', 'Kremlin', 'political summit', 'military parade', 'state meeting', 'diplomatic event'],
        sport: null,
        party: 'United Russia',
        opponents: ['NATO', 'United States', 'Ukraine', 'Western allies'],
    },
    'xi jinping': {
        display: 'Xi Jinping',
        roles: ['paramount leader', 'president', 'politician', 'general secretary'],
        organizations: ['Chinese Communist Party', 'People\'s Republic of China', 'Chinese Government'],
        typicalSettings: ['press conference', 'Great Hall of the People', 'political summit', 'state meeting', 'diplomatic event', 'party congress'],
        sport: null,
        party: 'Chinese Communist Party',
        opponents: ['United States', 'Taiwan', 'democracy advocates'],
    },
    'pope francis': {
        display: 'Pope Francis',
        roles: ['pope', 'religious leader', 'head of the Catholic Church', 'spiritual leader'],
        organizations: ['Vatican City', 'Catholic Church', 'Holy See'],
        typicalSettings: ['Vatican', 'St. Peter\'s Basilica', 'papal audience', 'religious ceremony', 'public mass', 'diplomatic meeting'],
        sport: null,
        party: null,
        opponents: [],
    },
    'kamala harris': {
        display: 'Kamala Harris',
        roles: ['vice president', 'politician', 'attorney', 'former senator'],
        organizations: ['White House', 'Democratic Party', 'US Senate', 'US Department of Justice'],
        typicalSettings: ['press conference', 'White House', 'political rally', 'political event', 'Senate chamber', 'campaign event'],
        sport: null,
        party: 'Democratic',
        opponents: ['Republican Party', 'Donald Trump'],
    },
    'kim kardashian': {
        display: 'Kim Kardashian',
        roles: ['media personality', 'socialite', 'businesswoman', 'reality TV star'],
        organizations: ['Kardashian family', 'SKIMS', 'KKW Beauty'],
        typicalSettings: ['red carpet', 'fashion event', 'TV studio', 'photoshoot', 'courtroom', 'social media post'],
        sport: null,
        party: null,
        opponents: [],
    },
    'mark zuckerberg': {
        display: 'Mark Zuckerberg',
        roles: ['CEO', 'businessman', 'programmer', 'tech executive'],
        organizations: ['Meta', 'Facebook', 'Instagram', 'WhatsApp'],
        typicalSettings: ['tech conference', 'press conference', 'office', 'congressional hearing', 'developer conference'],
        sport: null,
        party: null,
        opponents: [],
    },
    'jeff bezos': {
        display: 'Jeff Bezos',
        roles: ['CEO', 'businessman', 'entrepreneur', 'founder'],
        organizations: ['Amazon', 'Blue Origin', 'The Washington Post'],
        typicalSettings: ['tech conference', 'press conference', 'office', 'warehouse', 'rocket launch', 'product launch'],
        sport: null,
        party: null,
        opponents: [],
    },
    'lebron james': {
        display: 'LeBron James',
        roles: ['basketball player', 'athlete', 'NBA star', 'entrepreneur'],
        organizations: ['Los Angeles Lakers', 'NBA', 'LeBron James Family Foundation'],
        typicalSettings: ['basketball court', 'NBA game', 'press conference', 'arena', 'sports award ceremony'],
        sport: 'basketball',
        party: null,
        opponents: [],
    },
    'serena williams': {
        display: 'Serena Williams',
        roles: ['tennis player', 'athlete', 'champion', 'entrepreneur'],
        organizations: ['WTA', 'US Open', 'Wimbledon'],
        typicalSettings: ['tennis court', 'Grand Slam match', 'press conference', 'award ceremony', 'sports event'],
        sport: 'tennis',
        party: null,
        opponents: [],
    },
    'roger federer': {
        display: 'Roger Federer',
        roles: ['tennis player', 'athlete', 'champion', 'former world No. 1'],
        organizations: ['ATP', 'Wimbledon', 'Laver Cup'],
        typicalSettings: ['tennis court', 'Grand Slam match', 'press conference', 'award ceremony', 'sports event'],
        sport: 'tennis',
        party: null,
        opponents: [],
    },
    'leonardo dicaprio': {
        display: 'Leonardo DiCaprio',
        roles: ['actor', 'film producer', 'environmental activist'],
        organizations: [],
        typicalSettings: ['movie premiere', 'red carpet', 'film set', 'awards ceremony', 'interview', 'environmental summit'],
        sport: null,
        party: null,
        opponents: [],
    },
    'keanu reeves': {
        display: 'Keanu Reeves',
        roles: ['actor', 'film producer', 'musician'],
        organizations: [],
        typicalSettings: ['movie premiere', 'red carpet', 'film set', 'awards ceremony', 'interview'],
        sport: null,
        party: null,
        opponents: [],
    },
    'oprah winfrey': {
        display: 'Oprah Winfrey',
        roles: ['talk show host', 'media executive', 'philanthropist', 'producer'],
        organizations: ['OWN', 'Oprah Winfrey Network', 'Harpo Productions', 'O Magazine'],
        typicalSettings: ['TV studio', 'talk show set', 'press conference', 'philanthropy event', 'award ceremony'],
        sport: null,
        party: null,
        opponents: [],
    },
};

/**
 * Get context info for a known celebrity.
 * @param {string} name - The name as returned by face_api.py (lowercase, space-separated).
 * @returns {object|null} The celebrity context entry, or null if not found.
 */
export function getCelebrityContext(name) {
    if (!name) return null;
    const key = name.toLowerCase().trim();
    return CELEBRITY_CONTEXTS[key] || null;
}

const TEAM_ALIAS_ENTRIES = [
    ['paris saint-germain', 'PSG'],
    ['borussia dortmund', 'Borussia Dortmund'],
    ['manchester united', 'Manchester United'],
    ['manchester city', 'Manchester City'],
    ['atletico madrid', 'Atlético Madrid'],
    ['internazionale', 'Inter Milan'],
    ['inter miami', 'Inter Miami'],
    ['inter milan', 'Inter Milan'],
    ['real madrid', 'Real Madrid'],
    ['bayern munich', 'Bayern Munich'],
    ['sporting cp', 'Sporting CP'],
    ['al ittihad', 'Al Ittihad'],
    ['man united', 'Manchester United'],
    ['man city', 'Manchester City'],
    ['barcelona', 'FC Barcelona'],
    ['tottenham', 'Tottenham'],
    ['ac milan', 'AC Milan'],
    ['al nassr', 'Al Nassr'],
    ['al hilal', 'Al Hilal'],
    ['al-hilal', 'Al Hilal'],
    ['paris sg', 'PSG'],
    ['liverpool', 'Liverpool'],
    ['dortmund', 'Borussia Dortmund'],
    ['flamengo', 'Flamengo'],
    ['atletico', 'Atlético Madrid'],
    ['sporting', 'Sporting CP'],
    ['man utd', 'Manchester United'],
    ['chelsea', 'Chelsea'],
    ['arsenal', 'Arsenal'],
    ['juventus', 'Juventus'],
    ['benfica', 'Benfica'],
    ['bayern', 'Bayern Munich'],
    ['napoli', 'Napoli'],
    ['santos', 'Santos'],
    ['milan', 'AC Milan'],
    ['inter', 'Inter Milan'],
    ['barca', 'FC Barcelona'],
    ['spurs', 'Tottenham'],
    ['porto', 'Porto'],
    ['juve', 'Juventus'],
    ['roma', 'AS Roma'],
    ['ajax', 'Ajax'],
    ['psg', 'PSG'],
    ['bvb', 'Borussia Dortmund'],
    ['fcb', 'FC Barcelona'],
];

/**
 * Find any known team names mentioned in text.
 * Returns the display name of the first matched team, or null.
 */
export function findTeamInText(text) {
    if (!text) return null;
    const d = text.toLowerCase();
    for (const [alias, display] of TEAM_ALIAS_ENTRIES) {
        if (d.includes(alias)) return display;
    }
    return null;
}

function normalizeTeam(s) {
    return s.toLowerCase()
        .replace(/['']/g, '')
        .replace(/[-_]/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
}

export function detectJerseyMismatch(text) {
    if (!text) return null;
    const d = text.toLowerCase();

    const playerEntries = Object.entries(PLAYER_CLUBS)
        .sort((a, b) => b[0].length - a[0].length);
    let matchedPlayer = null;
    let matchedPlayerDisplay = null;
    for (const [alias, clubs] of playerEntries) {
        if (d.includes(alias)) {
            matchedPlayer = alias;
            matchedPlayerDisplay = alias.split(' ').map(w => w[0].toUpperCase() + w.slice(1)).join(' ');
            break;
        }
    }
    if (!matchedPlayer) {
        return null;
    }

    let jerseyTeamDisplay = null;
    let jerseyTeamNorm = null;
    for (const [alias, display] of TEAM_ALIAS_ENTRIES) {
        if (d.includes(alias)) {
            jerseyTeamDisplay = display;
            jerseyTeamNorm = normalizeTeam(display);
            break;
        }
    }
    if (!jerseyTeamDisplay) {
        return null;
    }

    const knownClubs = PLAYER_CLUBS[matchedPlayer];
    const playedThere = knownClubs.some(c => {
        const cn = normalizeTeam(c);
        return cn === jerseyTeamNorm ||
            jerseyTeamNorm.includes(cn) ||
            cn.includes(jerseyTeamNorm);
    });

    if (playedThere) {
        return null;
    }

    const knownDisplay = knownClubs.map(c =>
        c.split(' ').map(w => w[0].toUpperCase() + w.slice(1)).join(' ')
    ).join(', ');
    return {
        player: matchedPlayerDisplay,
        jerseyTeam: jerseyTeamDisplay,
        knownClubs: knownDisplay,
        mismatchMsg: `${matchedPlayerDisplay} has NEVER played for ${jerseyTeamDisplay}. Known clubs: ${knownDisplay}.`,
    };
}
