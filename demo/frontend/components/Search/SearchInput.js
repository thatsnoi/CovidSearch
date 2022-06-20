import { FiSearch } from 'react-icons/fi'

export default function SearchInput() {
  return (
    <div className="relative w-full h-10">
      <input
        className="absolute top-0 left-0 right-0 bottom-0
         bg-white rounded-full border-2 border-indigo-300 px-4 flex items-center"
        placeholder="Type to Search..."
      ></input>
      <div className="absolute flex items-center top-0 bottom-0 right-5 cursor-pointer">
        <FiSearch />
      </div>
    </div>
  )
}
