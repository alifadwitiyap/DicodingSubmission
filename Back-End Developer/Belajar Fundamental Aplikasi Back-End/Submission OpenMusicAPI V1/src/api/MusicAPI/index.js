const albumhandler = require('./albumHandler')
const songhandler = require('./songHandler')
const routes = require('./routes')

module.exports = {
    name: 'openmusic API',
    version: '1.0.0',
    register: async (server, { albumService,songService,optionalService,validator }) => {
        const albumHandler = new albumhandler(albumService,optionalService,validator)
        const songHandler = new songhandler(songService,optionalService,validator)
        server.route(routes(albumHandler,songHandler))
    }
}