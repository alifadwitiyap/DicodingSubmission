
const routes = (albumHandler,songHandler) => [
    {
        method: 'POST',
        path: '/albums',
        handler: albumHandler.postAlbumHandler
    }, {
        method: 'GET',
        path: '/albums/{id}',
        handler: albumHandler.getAlbumByIdHandler
    }, {
        method: 'PUT',
        path: '/albums/{id}',
        handler: albumHandler.putAlbumByIdHandler
    }, {
        method: 'DELETE',
        path: '/albums/{id}',
        handler: albumHandler.deleteAlbumByIdHandler
    },{
        method: 'POST',
        path: '/songs',
        handler: songHandler.postSongHandler
    },{
        method: 'GET',
        path: '/songs',
        handler: songHandler.getSongs
    },{
        method: 'GET',
        path: '/songs/{id}',
        handler: songHandler.getSongById        
    },{
        method: 'PUT',
        path: '/songs/{id}',
        handler: songHandler.updateSongById   
    },{
        method: 'DELETE',
        path: '/songs/{id}',
        handler: songHandler.deleteSongById 
    }
]

module.exports = routes