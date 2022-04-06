const invariantError = require('../exception/InvariantError');
const schema = require('./schema')

module.exports = (payload, marker) => {
    let validationResult
    if (marker === 'album') {
        validationResult = schema.albumSchema.validate(payload)
    } else if (marker === 'song') {
        validationResult = schema.songSchema.validate(payload)
    } else {
        throw new invariantError('invalid marker')
    }
    if (validationResult.error) {
        throw new invariantError(validationResult.error.message)
    }
}



