const joi = require('joi');

module.exports = {
    albumSchema:joi.object({
        name:joi.string().required(),
        year:joi.number().required()
    }),
    songSchema:joi.object({
        title:joi.string().required(),
        year:joi.number().required(),
        genre:joi.string().required(),
        performer:joi.string().required(),
        duration:joi.number(),
        albumId:joi.string()
    })
}