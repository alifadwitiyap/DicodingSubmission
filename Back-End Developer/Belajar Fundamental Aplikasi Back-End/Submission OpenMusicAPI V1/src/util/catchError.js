const ClientError = require('../exception/ClientError');

module.exports=async (error,h)=>{
    if (error instanceof ClientError) {
        return h.response({
            status: 'fail',
            message: error.message
        }).code(error.statusCode)
    }
    return h.response({
        status: 'error',
        message: "Maaf, terjadi kegagalan pada server kami"
    }).code(500)
}