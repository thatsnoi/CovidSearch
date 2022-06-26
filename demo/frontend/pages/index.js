import Link from 'next/link'
import { useRouter } from 'next/router'
import { FiGithub } from 'react-icons/fi'
import SearchInput from '../components/Search/SearchInput'

export default function Home() {
  const router = useRouter()

  const search = (query) => {
    router.push('results?query=' + query)
  }

  return (
    <div
      className="flex relative justify-center items-center bg-indigo-900"
      style={{ height: '50vh' }}
    >
      <FiGithub className="absolute top-4 right-4 text-white text-2xl cursor-pointer" />
      <div className="flex flex-col items-center space-y-5 w-1/2">
        <Link href="/">
          <h1 className="text-white font-sans text-5xl font-semibold cursor-pointer">
            Covid<span className="text-indigo-200">Search</span>
          </h1>
        </Link>
        <SearchInput search={search} />
      </div>
    </div>
  )
}
