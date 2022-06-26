import { FiSearch } from 'react-icons/fi'
import { useState, useEffect } from 'react'

export default function SearchInput({ search, value }) {
  const [query, setQuery] = useState('')

  const handleChange = (event) => {
    setQuery(event.target.value)
  }

  useEffect(() => {
    setQuery(value)
  }, [value])

  return (
    <form
      className="relative w-full h-10"
      onSubmit={(event) => {
        event.preventDefault()
        search(query)
      }}
    >
      <input
        className="absolute top-0 left-0 right-0 bottom-0
         bg-white rounded-full border-2 border-indigo-300 px-4 flex items-center pr-10"
        placeholder="Type to Search..."
        onChange={handleChange}
        value={query}
      ></input>
      <div className="absolute flex items-center top-0 bottom-0 right-5 cursor-pointer">
        <button type="submit">
          <FiSearch size={22} />
        </button>
      </div>
    </form>
  )
}
