const {simpanBuku,tampilkanSeluruhBuku,tampilkanDetailBuku,updateBuku,deleteBuku} = require('./handler')

const route =[
{
    method:`POST`,
    path:`/books`,
    handler:simpanBuku
}
,{
    method:`GET`,
    path:`/books`,
    handler:tampilkanSeluruhBuku
}
,{
    method:`GET`,
    path:`/books/{bookId}`,
    handler:tampilkanDetailBuku
}
,{
    method:`PUT`,
    path:`/books/{bookId}`,
    handler:updateBuku
}
,{
    method:`DELETE`,
    path:`/books/{bookId}`,
    handler:deleteBuku
}
]

module.exports = route