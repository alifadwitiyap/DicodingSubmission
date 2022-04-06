const { Pool } = require('pg')
const getSongsMapper = require('../util/getSongsMapper')



class optionalService {

    constructor(){
        this._db= new Pool
    }

    async getSongsByAlbumId({id}){
        const query = {
            text: "SELECT * FROM song WHERE album_id = $1",
            values: [id]
        }
        const result = await this._db.query(query)
        return result.rows.map(getSongsMapper)
    }
}

module.exports = optionalService