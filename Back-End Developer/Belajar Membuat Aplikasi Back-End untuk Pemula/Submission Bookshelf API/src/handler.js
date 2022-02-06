const { nanoid } = require('nanoid')
const books = require('./bookshelf')

const simpanBuku = (req, h) => {
    const { name, year, author, summary, publisher, pageCount, readPage, reading } = req.payload
    const id = nanoid(16)
    const finished = pageCount == readPage
    const insertedAt = new Date().toISOString()
    const updatedAt = insertedAt

    //kita filter dulu

    if (name === undefined) {
        return h.response({
            status: "fail",
            message: "Gagal menambahkan buku. Mohon isi nama buku"
        }).code(400)
    }
    if (readPage > pageCount) {
        return h.response({
            status: "fail",
            message: "Gagal menambahkan buku. readPage tidak boleh lebih besar dari pageCount"
        }).code(400)
    }


    newBooks = {
        id,
        name,
        year,
        author,
        summary,
        publisher,
        pageCount,
        readPage,
        finished,
        reading,
        insertedAt,
        updatedAt
    }

    books.push(newBooks)


    const isSuccess = books.filter((books) => books.id === id).length > 0;

    if (isSuccess) {
        return h.response({
            status: `success`,
            message: `Buku berhasil ditambahkan`,
            data: {
                bookId: id
            }
        }).code(201)
    } else {
        return h.response({
            status: `error`,
            message: `Buku gagal ditambahkan`
        }).code(500)
    }


}

const tampilkanSeluruhBuku = (req,h) => {

    // di filter dulu
    const query =req.query
    let filteredBook=[...books]
    
    if (Object.keys(query).length !== 0 && Object.keys(query)[0]!=="name" ){
        filteredBook= filteredBook.filter((b)=>b[Object.keys(query)]==Object.values(query))
    }else if (Object.keys(query).length !== 0 && Object.keys(query)[0]==="name" ){
        temp=[...filteredBook]
        filteredBook=[]
        for (i of temp){

            console.log(i);
            let nama=i.name.toLowerCase()
            if (nama.search("dicoding") !== -1 ){
                filteredBook.push(i)
            }
        }

    }
    

    return {
        status: "success",
        data: {
            books: filteredBook.map((item) => {
                return {
                    id: item.id,
                    name: item.name,
                    publisher: item.publisher
                }
            }
            )
        }
    }
}

const tampilkanDetailBuku = (req, h) => {
    const { bookId } = req.params

    const index = books.findIndex((books) => books.id == bookId)


    if (index !== -1) {
        return {
            status: "success",
            data: {
                book:books[index]
            }
        }
    } else {
        return h.response({
            status: "fail",
            message: "Buku tidak ditemukan"
        }).code(404)
    }
}

const updateBuku = (req, h) => {
    const { name, year, author, summary, publisher, pageCount, readPage, reading } = req.payload
    const { bookId } = req.params
    const updateAt = new Date().toISOString()
    const finished = pageCount == readPage

    //kita filter dulu
    if (name === undefined) {
        return h.response({
            status: "fail",
            message: "Gagal memperbarui buku. Mohon isi nama buku"
        }).code(400)
    }
    if (readPage > pageCount) {
        return h.response({
            status: "fail",
            message: "Gagal memperbarui buku. readPage tidak boleh lebih besar dari pageCount"
        }).code(400)
    }

    const index = books.findIndex((books) => books.id == bookId)


    if (index !== -1) {
        books[index] = {
            ...books[index],
            name,
            year,
            author,
            summary,
            publisher,
            pageCount,
            readPage,
            reading,
            updateAt,
            finished
        }

        return {
            "status": "success",
            "message": "Buku berhasil diperbarui"
        }
    } else {
        return h.response({
            "status": "fail",
            "message": "Gagal memperbarui buku. Id tidak ditemukan"
        }).code(404)
    }

}

const deleteBuku = (req, h) => {
    const { bookId } = req.params
    const index = books.findIndex((books) => books.id == bookId)


    if (index !== -1) {
        books.splice(index, 1)
        return {
            "status": "success",
            "message": "Buku berhasil dihapus"
        }

    } else {
        return h.response({
            "status": "fail",
            "message": "Buku gagal dihapus. Id tidak ditemukan"
        }).code(404)
    }
}





module.exports = { simpanBuku, tampilkanSeluruhBuku, tampilkanDetailBuku, updateBuku, deleteBuku }