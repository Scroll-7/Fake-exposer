const PLAYER_CLUBS = {
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
    'michael olise': ['bayern munich', 'crystal palace', 'reading'],
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
