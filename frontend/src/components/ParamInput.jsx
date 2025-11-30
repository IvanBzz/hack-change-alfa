import React from 'react';
import { formatFeatureName } from '../utils';

const ParamInput = React.memo(({ paramKey, value, onChange }) => {
  return (
    <div className="flex flex-col gap-1 p-2 hover:bg-white rounded-lg transition-colors">
       <label className="text-xs font-bold text-gray-500 truncate uppercase tracking-wide" title={formatFeatureName(paramKey)}>
         {formatFeatureName(paramKey)}
       </label>
       {typeof value === 'number' ? (
          <input 
            type="number" 
            className="p-2 border border-gray-200 rounded-lg text-sm focus:border-blue-500 outline-none bg-white"
            value={value}
            onChange={(e) => onChange(paramKey, parseFloat(e.target.value))}
          />
       ) : (
          <input 
            type="text" 
            className="p-2 border border-gray-200 rounded-lg text-sm focus:border-blue-500 outline-none bg-white"
            value={value}
            onChange={(e) => onChange(paramKey, e.target.value)}
          />
       )}
    </div>
  );
}, (prevProps, nextProps) => {
  return prevProps.value === nextProps.value && prevProps.paramKey === nextProps.paramKey;
});

export default ParamInput;
