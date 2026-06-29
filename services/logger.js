const LEVELS = { ERROR: 0, WARN: 1, INFO: 2, DEBUG: 3 };
const current = LEVELS[(process.env.LOG_LEVEL || 'INFO').toUpperCase()] ?? LEVELS.INFO;

function format(level, args) {
    const ts = new Date().toISOString();
    return [`[${ts}] [${level}]`, ...args];
}

export const logger = {
    error: (...args) => { if (LEVELS.ERROR <= current) process.stderr.write(format('ERROR', args).join(' ') + '\n'); },
    warn: (...args) => { if (LEVELS.WARN <= current) process.stderr.write(format('WARN', args).join(' ') + '\n'); },
    info: (...args) => { if (LEVELS.INFO <= current) process.stdout.write(format('INFO', args).join(' ') + '\n'); },
    debug: (...args) => { if (LEVELS.DEBUG <= current) process.stdout.write(format('DEBUG', args).join(' ') + '\n'); },
};
