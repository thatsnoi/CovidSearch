import Link from 'next/link'

export default function Result({ data }) {
  return (
    <div className="flex flex-col">
      {data.url.length > 0 ? (
        <a href={data.url.split(';')[0]} target="_blank" rel="noreferrer">
          <h2 className="font-semibold text-xl cursor-pointer hover:underline">
            {data.title}
          </h2>
        </a>
      ) : (
        <h2 className="font-semibold text-xl">{data.title}</h2>
      )}
      <p className="font-semibold text-indigo-900 line-clamp-1">
        {data.authors}
      </p>
      <p className="text-sm line-clamp-4 pt-2">
        {data.abstract.length > 0 ? data.abstract : 'No abstract available.'}
      </p>
      <p className="text-gray-600 text-sm pt-1">
        <span className="font-semibold">{data.source_x}</span>
        {' ' + data.publish_time}
      </p>
    </div>
  )
}
