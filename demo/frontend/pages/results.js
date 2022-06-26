import SearchInput from '../components/Search/SearchInput'
import { FiGithub } from 'react-icons/fi'
import Result from '../components/Results/Result'
import axios from 'axios'
import { useState } from 'react'
import Settings from '../components/Results/Settings'
import Link from 'next/link'
import { useRouter } from 'next/router'
import { useEffect } from 'react'

export default function Results() {
  const router = useRouter()
  const routerQuery = router.query.query

  useEffect(() => {
    if (routerQuery != undefined && routerQuery.length > 0) {
      search(routerQuery)
    }
  }, [routerQuery, search])

  const [results, setResults] = useState([])
  const [settings, setSettings] = useState({
    ce: false,
    top_k: 20,
    fuse: false,
  })
  const [error, setError] = useState(false)
  const [loading, setLoading] = useState(false)

  async function search(query) {
    setLoading(true)
    try {
      const res = await axios.post('http://127.0.0.1:8000/search', {
        query: query,
        ...settings,
      })
      console.log(res)
      setResults(res.data)
      setLoading(false)
    } catch (error) {
      setError(true)
      setLoading(false)
    }
  }

  function updateSettings(newSettings) {
    setSettings(newSettings)
  }

  return (
    <div className="flex flex-col relative h-screen overflow-hidden">
      <div className="flex justify-between items-center p-4 bg-indigo-900 h-20 w-full">
        <div className="w-2/3 flex items-center space-x-10">
          <Link href="/">
            <h1 className="text-white font-sans text-3xl font-semibold cursor-pointer">
              Covid<span className="text-indigo-200">Search</span>
            </h1>
          </Link>
          <SearchInput search={search} value={routerQuery} />
        </div>
        <FiGithub className="text-white text-2xl cursor-pointer" />
      </div>
      <div className="flex h-full">
        <div className="h-full bg-gray-200 p-5 whitespace-nowrap">
          <h2 className="text-2xl font-semibold pb-5">Settings</h2>
          <Settings updateSettings={updateSettings} />
        </div>
        <div className="px-10 py-5 space-y-5 overflow-y-scroll">
          {loading
            ? 'Loading...'
            : error
            ? 'An error has occured.'
            : results.map((result) => {
                return (
                  <Result
                    key={result.id}
                    title={result.title}
                    text={result.text}
                  />
                )
              })}
        </div>
      </div>
    </div>
  )
}
