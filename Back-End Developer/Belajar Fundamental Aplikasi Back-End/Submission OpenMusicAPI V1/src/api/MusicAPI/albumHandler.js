const catchError = require('../../util/catchError')

class albumHandler {

    constructor(albumService, optionalService, validator) {
        this._albumService = albumService
        this._optionalService = optionalService
        this._validator = validator
        this.postAlbumHandler = this.postAlbumHandler.bind(this)
        this.getAlbumByIdHandler = this.getAlbumByIdHandler.bind(this)
        this.putAlbumByIdHandler = this.putAlbumByIdHandler.bind(this)
        this.deleteAlbumByIdHandler = this.deleteAlbumByIdHandler.bind(this)
    }

    async postAlbumHandler(request, h) {
        try {
            this._validator(request.payload, 'album')
            const albumId = await this._albumService.addAlbum(request.payload)
            return h.response({
                status: 'success',
                data: { albumId }
            }).code(201)

        } catch (error) {
            return catchError(error, h)
        }
    }

    async getAlbumByIdHandler(request, h) {
        try {
            let album = await this._albumService.getAlbumByID(request.params)
            album.songs = await this._optionalService.getSongsByAlbumId(album)
            return h.response({
                status: 'success',
                data: {album}
            }).code(200)
        } catch (error) {
            return catchError(error, h)
        }
    }

    async putAlbumByIdHandler(request, h) {
        try {
            this._validator(request.payload, 'album')
            await this._albumService.updateAlbumByID(request.params, request.payload)
            return h.response({
                status: 'success',
                message: 'Data Berhasil di Update'
            }).code(200)
        } catch (error) {
            return catchError(error, h)
        }
    }

    async deleteAlbumByIdHandler(request, h) {
        try {
            await this._albumService.deleteAlbumByID(request.params)
            return h.response({
                status: 'success',
                message: 'Data Berhasil di Hapus'
            }).code(200)
        } catch (error) {
            return catchError(error, h)
        }
    }


}

module.exports = albumHandler
