const { Pool } = require('pg')
const { nanoid } = require('nanoid')
const NotFoundError = require('../exception/NotFoundError');
const InvariantError = require('../exception/InvariantError');

class AlbumService {
    constructor() {
        this._db = new Pool()
    }

    async addAlbum({ name, year }) {
        const id = "album-" + nanoid(16)
        const query = {
            text: "INSERT INTO album VALUES ($1,$2,$3) RETURNING id",
            values: [id, name, year]
        }
        const result = await this._db.query(query)
        if (result.rows.length < 1) {
            throw new InvariantError("Album gagal ditambahkan")
        }
        return result.rows[0].id
    }

    async getAlbumByID({ id }) {
        const query = {
            text: "SELECT * FROM album WHERE id = $1",
            values: [id]
        }
        const result = await this._db.query(query)
        if (result.rows.length < 1) {
            throw new NotFoundError("Album tidak ditemukan")
        }
        return result.rows[0]
    }

    async updateAlbumByID({ id }, { name, year }) {
        const query = {
            text: "UPDATE album SET name=$1, year=$2 WHERE id=$3 RETURNING id",
            values: [name, year, id]
        }
        const result = await this._db.query(query)
        if (result.rows.length < 1) {
            throw new NotFoundError("Album tidak ditemukan")
        }
    }


    async deleteAlbumByID({ id }) {
        const query = {
            text: "DELETE FROM album WHERE id=$1 RETURNING id",
            values: [id]
        }
        const result = await this._db.query(query)
        if (result.rows.length < 1) {
            throw new NotFoundError("Album tidak ditemukan")
        }
    }
}

module.exports = AlbumService