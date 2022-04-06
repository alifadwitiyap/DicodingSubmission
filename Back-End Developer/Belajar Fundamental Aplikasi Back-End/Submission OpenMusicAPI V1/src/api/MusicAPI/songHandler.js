const catchError = require('../../util/catchError')

class SongHandler {

    constructor(songService, optionalService, validator) {
        this._songService = songService
        this._optionalService = optionalService
        this._validator = validator
        this.postSongHandler = this.postSongHandler.bind(this)
        this.getSongs = this.getSongs.bind(this)
        this.getSongById = this.getSongById.bind(this)
        this.updateSongById = this.updateSongById.bind(this)
        this.deleteSongById = this.deleteSongById.bind(this)
    }

    async postSongHandler(request, h) {
        try {
            this._validator(request.payload, 'song')
            const songId = await this._songService.addSong(request.payload)
            return h.response({
                status: 'success',
                data: { songId }
            }).code(201)
        } catch (error) {
            return catchError(error, h)
        }
    }

    async getSongs(request, h) {
        try {
            const songs = await this._songService.getSongs(request.query)
            return h.response({
                status: 'success',
                data: { songs }
            }).code(200)
        } catch (error) {
            return catchError(error, h)
        }
    }

    async getSongById(request, h) {
        try {
            const song = await this._songService.getSongById(request.params)
            return h.response({
                status: 'success',
                data: { song }
            }).code(200)
        } catch (error) {
            return catchError(error, h)
        }
    }

    async updateSongById(request, h) {
        try {
            this._validator(request.payload, 'song')
            await this._songService.updateSongById(request.params, request.payload)
            return h.response({
                status: 'success',
                message: "Lagu berhasil di updated"
            }).code(200)
        } catch (error) {
            return catchError(error, h)
        }
    }

    async deleteSongById(request, h) {
        try {
            await this._songService.deleteSongByID(request.params)
            return h.response({
                status: 'success',
                message: "Lagu berhasil di delete"
            }).code(200)
        } catch (error) {
            return catchError(error, h)
        }
    }

}

module.exports = SongHandler

